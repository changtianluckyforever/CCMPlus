from modules import ccformer
from prediction import autoformer, dlinear, timesnet, lightts, patchtst
from modules import magic_scaler as magicscaler
from modules import hpa
from prediction.data_loader import data_provider
from prediction.tools import EarlyStopping, adjust_learning_rate, visual
from prediction.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np
from prediction.simulator_env import convert_resource


class Runner:
    def __init__(self, args):
        super(Runner, self).__init__()
        self.args = args
        self.model_dict = {
            "ccformer": ccformer,
            "magicscaler": magicscaler,
            "hpa": hpa,
            "timesnet": timesnet,
            "autoformer": autoformer,
            # 'Transformer': Transformer,
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            "dlinear": dlinear,
            # 'FEDformer': FEDformer,
            # 'Informer': Informer,
            "lightts": lightts,
            # 'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            "patchtst": patchtst,
            # 'Pyraformer': Pyraformer,
            # 'MICN': MICN,
            # 'Crossformer': Crossformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def valid(self, valid_loader, criterion):
        total_loss = []
        self.model.eval()
        self.args.model_training_flag = False
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                B, N, L, H = batch_x.shape

                if self.args.model != "ccformer":
                    batch_x_mark = batch_x_mark.repeat(1, N, 1, 1)
                    batch_y_mark = batch_y_mark.repeat(1, N, 1, 1)
                    batch_x_mark, batch_y_mark = batch_x_mark.flatten(0, 1), batch_y_mark.flatten(0, 1)
                    batch_x, batch_y = batch_x.flatten(0, 1), batch_y.flatten(0, 1)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == "magicscaler":
                        outputs, mll_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs.reshape(B, N, -1, H)
                batch_y = batch_y.reshape(B, N, -1, H)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[..., -self.args.pred_len :, f_dim:]
                batch_y = batch_y[..., -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                # if self.args.model == "magicscaler":
                #     loss = loss + mll_loss

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.args.current_epoch = epoch

            self.model.train()
            self.args.model_training_flag = True
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.args.current_iter = iter_count
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                B, N, L, H = batch_x.shape

                if self.args.model != "ccformer":
                    batch_x_mark = batch_x_mark.repeat(1, N, 1, 1)
                    batch_y_mark = batch_y_mark.repeat(1, N, 1, 1)
                    batch_x_mark, batch_y_mark = batch_x_mark.flatten(0, 1), batch_y_mark.flatten(0, 1)
                    batch_x, batch_y = batch_x.flatten(0, 1), batch_y.flatten(0, 1)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == "magicscaler":
                        outputs, mll_loss = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # the shape of outputs is [B, N, args.pred_len]
                outputs = outputs.reshape(B, N, -1, H)
                batch_y = batch_y.reshape(B, N, -1, H)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[..., -self.args.pred_len :, f_dim:]
                batch_y = batch_y[..., -self.args.pred_len :, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                # if self.args.model == "magicscaler":
                #     loss = loss + mll_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.valid(vali_loader, criterion)
            test_loss = self.valid(test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(os.path.join("../checkpoints/" + setting, "checkpoint.pth")))

        preds = []
        trues = []
        folder_path = "../test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.args.model_training_flag = False
        with torch.no_grad():
            # *********
            testing_begin_time = time.time()
            testing_iter_count = 0
            # *********
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # *********
                testing_iter_count += 1
                # *********
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                B, N, L, H = batch_x.shape

                if self.args.model != "ccformer":
                    batch_x_mark = batch_x_mark.repeat(1, N, 1, 1)
                    batch_y_mark = batch_y_mark.repeat(1, N, 1, 1)
                    batch_x_mark, batch_y_mark = batch_x_mark.flatten(0, 1), batch_y_mark.flatten(0, 1)
                    batch_x, batch_y = batch_x.flatten(0, 1), batch_y.flatten(0, 1)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == "magicscaler":
                        outputs, mll_loss = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.model == "hpa":
                    outputs = batch_x[:, -self.args.pred_len :, :]
                    
                outputs = outputs.reshape(B, N, -1, H)
                batch_y = batch_y.reshape(B, N, -1, H)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[..., -self.args.pred_len :, f_dim:]
                batch_y = batch_y[..., -self.args.pred_len :, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs.squeeze(-1)
                true = batch_y.squeeze(-1)

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     print("the batch size for testing is: ", B)
                #     print("\ttesting_ iters: {0} | ".format(i + 1))
                #     speed = (time.time() - testing_begin_time) / testing_iter_count
                #     print("\tspeed: {:.4f}s/iter ".format(speed))
                #     input_x = batch_x.reshape(B, N, -1, H).detach().cpu().numpy()
                #     gt = np.concatenate((input_x.squeeze(-1), true), axis=-1)[0, 0, :]
                #     pd = np.concatenate((input_x.squeeze(-1), pred), axis=-1)[0, 0, :]
                #     visual(gt, pd, os.path.join(folder_path, str(i) + ".png"))

        preds = np.array(preds)
        # the shape of preds is [number of batch, B, N, args.pred_len]
        trues = np.array(trues)
        # the shape of trues is [number of batch, B, N, args.pred_len]
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # the shape of preds is [number of batch * B, N, args.pred_len]
        # the shape of preds is [test_set_length, N, args.pred_len]
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # the shape of trues is [number of batch * B, N, args.pred_len]
        # the shape of trues is [test_set_length, N, args.pred_len]
        print("test shape:", preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}".format(mse, mae, rmse, mape, mspe))
        f = open(folder_path + "result_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}".format(mse, mae, rmse, mape, mspe))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        scale_test_set_length, scale_N, scale_pred_len = trues.shape
        # add scale back here for test_set if the self.scale is true
        # scale back with standard scaler
        if test_data.scale:
            # Reshape preds and trues for scaling back
            # Reshape to [test_set_length * pred_len, N] so that we can scale along dim=1 (N)
            preds_reshaped = preds.transpose(0, 2, 1).reshape(-1, scale_N)
            trues_reshaped = trues.transpose(0, 2, 1).reshape(-1, scale_N)

            # Apply inverse_transform to scale back along dim=1 (N)
            preds_scaled_back = test_data.inverse_transform(preds_reshaped)
            trues_scaled_back = test_data.inverse_transform(trues_reshaped)
            # the shape of preds_scaled_back is [test_set_length * pred_len, N]

            # Reshape back to original shape [test_set_length, N, args.pred_len]
            preds = preds_scaled_back.reshape(scale_test_set_length, scale_pred_len, scale_N).transpose(0, 2, 1)
            trues = trues_scaled_back.reshape(scale_test_set_length, scale_pred_len, scale_N).transpose(0, 2, 1)
            # the shape of trues  is [test_set_length, N, args.pred_len]
        np.save(folder_path + "pred.npy", preds.astype(float))
        np.save(folder_path + "true.npy", trues.astype(float))

        # simulated evaluation
        # 调用模拟评估函数, 在这个位置写
        final_latency, final_utility = convert_resource(folder_path)
        print("final_latency:{}, final_utility:{}".format(final_latency, final_utility))
        f = open(folder_path + "result_forecast.txt", "a")
        f.write("final_latency:{}, final_utility:{}".format(final_latency, final_utility))
        f.write("\n")
        f.write("\n")
        f.close()
