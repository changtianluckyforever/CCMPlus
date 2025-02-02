import datetime


def parse_trace(file_list, saved_fn):
    ts = []
    for fn in file_list:
        with open(fn, "r", encoding="latin-1") as f:
            for r in f.readlines():
                try:
                    split = r.split(" [")[1]
                    split = split.split("] \"")[0]
                    # dat, region_code = r.split(" ")
                    strp = datetime.datetime.strptime(split, '%d/%b/%Y:%H:%M:%S %z')
                    ts.append(strp)
                except Exception:
                    print(r)
    ts.sort()
    with open(f"./internet_traffic_archive/time_series_requests/{saved_fn}.tsv", "w", encoding="utf-8") as f:
        f.writelines("\n".join([datetime.datetime.strftime(t, "%Y/%m/%d-%H:%M:%S %z") for t in ts]))


file1 = "./internet_traffic_archive/clarknet_access_log_Aug28"
file2 = "./internet_traffic_archive/clarknet_access_log_Sep4"

file3 = "./internet_traffic_archive/calgary_access_log"

file4 = "./internet_traffic_archive/NASA_access_log_Aug95"
file5 = "./internet_traffic_archive/NASA_access_log_Jul95"

file6 = "./internet_traffic_archive/usask_access_log"
parse_trace([file6], "usask")
