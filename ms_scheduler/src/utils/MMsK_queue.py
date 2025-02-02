# https://github.com/CJP0/M-M-s-K_Queue_Simulation/blob/master/MMSK/mathAnalysis.py
import math


class MMSKQueue:
    def __init__(self, lam, mu, servers, buf_size, timeout, service_level):
        self.lam = lam
        self.mu = mu
        self.servers = servers
        self.buf_size = buf_size
        self.timeout = timeout
        self.service_level = service_level
        self.queue_capacity = buf_size - servers

        self._p0_v = None
        self._lq_v = None
        self._l_v = None
        self._lambda_eff_v = None

    def __str__(self):
        return f"lam\tmu\tservers\tcapacity\ttimeout\tservice_level\tqueue_capacity\n" \
               f"{self.lam},\t{self.mu},\t{self.servers},\t{self.buf_size},\t{self.timeout},\t{self.service_level},\t" \
               f"{self.queue_capacity}"

    @property
    def _p0(self):
        if self._p0_v is not None:
            return self._p0_v
        sum_t = 1.0
        for i in range(1, self.servers):
            sum_t += (pow((self.lam / self.mu), i) / math.factorial(i))
        sum_t += ((pow((self.lam / self.mu), self.servers) / math.factorial(self.servers)) * (
                (1 - pow((self.lam / (self.servers * self.mu)), (self.buf_size - self.servers + 1))) / (
                1 - (self.lam / (self.servers * self.mu)))))
        self._p0_v = 1 / sum_t
        return self._p0_v

    def _pn(self, n):
        if n < self.servers:
            return ((pow((self.lam / self.mu), n)) / math.factorial(n)) * self._p0
        t1 = pow((self.lam / self.mu), n)
        t2 = int(math.factorial(self.servers))
        t3 = int(pow(self.servers, (n - self.servers)))
        t4 = int(t2 * t3)
        return t1 / t4 * self._p0
        # return ((pow((self.lam / self.mu), n)) / (
        #     np.float64(np.math.factorial(self.servers) * np.power(self.servers, (n - self.servers)),
        #                dtype=np.float64))) * self._p0

    @property
    def _lq(self):
        if self._lq_v is not None:
            return self._lq_v
        result = 0
        for i in range(self.servers, (self.buf_size + 1)):
            result += ((i - self.servers) * self._pn(i))
        self._lq_v = result
        return self._lq_v

    @property
    def _l(self):
        if self._l_v is not None:
            return self._l_v
        sum1 = 0
        sum2 = 0
        for i in range(0, self.servers):
            temp = self._pn(i)
            sum1 += i * temp
            sum2 += temp
        sum2 = self.servers * (1 - sum2)
        self._l_v = self._lq + sum1 + sum2
        return self._l_v

    @property
    def _lambda_eff(self):
        if self._lambda_eff_v is not None:
            return self._lambda_eff_v
        self._lambda_eff_v = self.lam * (1 - self._pn(self.buf_size))
        return self._lambda_eff_v

    @property
    def _w(self):
        return self._l / self._lambda_eff

    @property
    def _wq(self):
        return self._lq / self._lambda_eff

    # ----------- service level ----------------

    @property
    def _util(self):
        p, q = 1, 1
        ll, lq = 0, 0
        for n in range(1, self.servers + 1):
            p = self.lam / (n * self.mu) * p
            q += p
            ll += n * p
        for n in range(self.servers + 1, self.servers + self.queue_capacity + 1):
            p = self.lam / (self.servers * self.mu) * p
            q += p
            ll += n * p
            lq += (n - self.servers) * p
        util_tmp = (ll - lq) / (q * self.servers)
        return util_tmp

    @property
    def _pr_wait(self):
        p, q = 1, 1
        if self.servers == 1:
            p = self.lam / self.mu * p
            q += p
            for n in (self.servers + 1, self.buf_size + 1):
                p = self.lam / (self.servers * self.mu) * p
                q += p
            return 1 - (1 / q)
        for n in range(1, self.servers + 1):
            p = self.lam / (n * self.mu) * p
            q += p
            if n == self.servers - 1 or self.servers == 1:
                Qtemp = q
        for n in range(self.servers + 1, self.buf_size + 1):
            p = self.lam / (self.mu * self.servers) * p
            q += p
        return 1 - Qtemp / q

    @property
    def _pr_balk(self):
        p, q = 1, 1
        for n in range(1, self.servers + 1):
            p = self.lam / (n * self.mu) * p
            q += p
        for n in range(self.servers + 1, self.buf_size + 1):
            p = self.lam / (self.mu * self.servers) * p
            q += p
        return p / q

    def _service_level(self, servers, queue_cap):
        r = 1
        sum1 = r
        for i in range(1, servers):
            r = (self.lam / (i * self.mu)) * r
            sum1 = sum1 + r
        a = servers * self.mu * self.timeout
        p = self.lam / (servers * self.mu)
        q = math.exp(-a)
        u = q
        sum2 = 0
        for i in range(1, queue_cap + 1):
            r = p * r
            sum2 = sum2 + r * u
            sum1 = sum1 + r
            q = (a / i) * q
            u = u + q
        return 1 - sum2 / sum1

    @property
    def _min_agents(self):
        exp_service_level = 1 - self.service_level
        servers = 1
        found_min = False
        while not found_min and servers <= self.buf_size:
            temp1 = 1 - self._service_level(servers, (self.buf_size - servers))
            if temp1 < exp_service_level:
                found_min = True
            else:
                servers += 1
        if servers > self.buf_size:
            return -1
        else:
            return servers

    @property
    def _min_servers(self):
        exp_service_level = 1 - self.service_level
        servers = 1
        found_min = False
        while not found_min:
            temp1 = 1 - self._service_level(servers, self.queue_capacity)
            if temp1 < exp_service_level:
                found_min = True
            else:
                servers += 1
        return servers

    def metrics(self):
        return {"p0": self._p0,
                "lq": self._lq,
                "l": self._l,
                "w": self._w,
                "wq": self._wq,
                "util": self._util,
                "pr_wait": self._pr_wait,
                "pr_balk": self._pr_balk,
                "service_level": self._service_level(self.servers, self.queue_capacity),
                "min_servers": self._min_servers,
                "min_agents": self._min_agents}


if __name__ == "__main__":
    que_obj = MMSKQueue(0.6936371527777778, 100.0, 20, 30, 0.1, 0.9)
    print(que_obj.metrics())
