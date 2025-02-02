import math

UtilMsg = "utilization ratio is larger than 1."


def util(arrival_rate, service_rate, servers, queue_capacity=None):
    """
    'calculates rho (server utilization)
    'the variable "Queue_Capacity" must be left as a Variant so
    'it can be passed as a parameter to the "IsMissing" function
    :param arrival_rate:
    :param service_rate:
    :param servers:
    :param queue_capacity:
    :return:
    """
    if queue_capacity is None:
        return arrival_rate / (service_rate * servers)
    p = 1
    q = 1
    ll = 0
    lq = 0
    for n in range(1, servers + 1):
        p = arrival_rate / (n * service_rate) * p
        q += p
        ll += n * p
    for n in range(servers + 1, servers + queue_capacity + 1):
        p = arrival_rate / (servers * service_rate) * p
        q += p
        ll += n * p
        lq += (n - servers) * p
    util_tmp = (ll - lq) / (q * servers)
    return util_tmp


def Po(arrival_rate, service_rate, servers, queue_capacity=None):
    """
    'calculates Po (the probability that the system is empty)
    'the variable "Queue_Capacity" must be left as a Variant so
    'it can be passed as a parameter to the "IsMissing" function
    :param arrival_rate:
    :param service_rate:
    :param servers:
    :param queue_capacity:
    :return:
    """
    sum1 = 0
    term = 1
    p = 1
    q = 1
    rho = arrival_rate / (service_rate * servers)
    if queue_capacity is None:
        if rho > 1:
            raise ValueError(UtilMsg)
        for n in range(servers):
            sum1 += term
            term *= arrival_rate / (service_rate * (n + 1))
        return 1 / (sum1 + (term / (1 - rho)))
    for n in range(1, servers + 1):
        p = arrival_rate / (n * service_rate) * p
        q += p
    for n in range(servers + 1, servers + queue_capacity + 1):
        p = arrival_rate / (service_rate * servers) * p
        q += p
    return 1 / q


def Lq(arrival_rate, service_rate, servers, queue_capacity=None):
    """
    'This function was developed according to calculations
    'by Armann Ingolfsson to calculate Lq (expected queue length)
    :param arrival_rate:
    :param service_rate:
    :param servers:
    :param queue_capacity:
    :return:
    """
    sum1 = 0
    term = 1
    if queue_capacity is None:
        rho = arrival_rate / (service_rate * servers)
        if rho > 1:
            raise ValueError(UtilMsg)
        for n in range(servers):
            sum1 += term
            term *= (arrival_rate / service_rate) / (n + 1)
        return (term * rho) / (math.pow((1 - rho), 2)) * (1 / (sum1 + (term / (1 - rho))))
    p = 1
    q = 1
    exp_q_length = 0
    for n in range(1, servers + 1):
        p = arrival_rate / (n * service_rate) * p
        q += p
    for n in range(servers + 1, servers + queue_capacity + 1):
        p = arrival_rate / (service_rate * servers) * p
        q += p
        exp_q_length += (n - servers) * p
    return exp_q_length / q


def L(arrival_rate, service_rate, servers, queue_capacity=None):
    """
    'calculates L (expected number in system)
    'the variable "Queue_Capacity" must be left as a Variant so
    'it can be passed as a parameter to the "IsMissing" function
    :param arrival_rate:
    :param service_rate:
    :param servers:
    :param queue_capacity:
    :return:
    """
    sum1 = 0
    term = 1
    if queue_capacity is None:
        rho = arrival_rate / (service_rate * servers)
        if rho > 1:
            raise ValueError(UtilMsg)
        for n in range(servers):
            sum1 += term
            term *= (arrival_rate / service_rate) / (n + 1)
        return (term * rho) / (math.pow((1 - rho), 2)) * (1 / (sum1 + (term / (1 - rho)))) + (
                arrival_rate / service_rate)
    p = 1
    q = 1
    exp_n_system = 0
    for n in range(1, servers + 1):
        p = arrival_rate / (n * service_rate) * p
        q += p
        exp_n_system += n * p
    for n in range(servers + 1, servers + queue_capacity + 1):
        p = arrival_rate / (service_rate * servers) * p
        q += p
        exp_n_system += n * p
    return exp_n_system / q


def Wq(arrival_rate, service_rate, servers, queue_capacity=None):
    """
    'calculates Wq (expected wait in queue)
    'the variable "Queue_Capacity" must be left as a Variant so
    'it can be passed as a parameter to the "IsMissing" function
    :param arrival_rate:
    :param service_rate:
    :param servers:
    :param queue_capacity:
    :return:
    """
    sum1 = 0
    term = 1
    if queue_capacity is None:
        rho = arrival_rate / (service_rate * servers)
        if rho > 1:
            raise ValueError(UtilMsg)
        for n in range(servers):
            sum1 += term
            term *= (arrival_rate / service_rate) / (n + 1)
        return (term * rho) / (math.pow((1 - rho), 2)) * (1 / (sum1 + (term / (1 - rho)))) / arrival_rate
    p = 1
    q = 1
    lq = 0
    for n in range(1, servers + 1):
        p = arrival_rate / (n * service_rate) * p
        q += p
    for n in range(servers + 1, servers + queue_capacity + 1):
        p = arrival_rate / (service_rate * servers) * p
        q += p
        lq = lq + (n - servers) * p
    lambda_bar = arrival_rate * (1 - p / q)
    return (lq / q) / lambda_bar


def W(arrival_rate, service_rate, servers, queue_capacity=None):
    """
    'calculates W (expected total time in system)
    'the variable "Queue_Capacity" must be left as a Variant so
    'it can be passed as a parameter to the "IsMissing" function
    :param arrival_rate:
    :param service_rate:
    :param servers:
    :param queue_capacity:
    :return:
    """
    sum1 = 0
    term = 1
    if queue_capacity is None:
        rho = arrival_rate / (service_rate * servers)
        if rho > 1:
            raise ValueError(UtilMsg)
        for n in range(servers):
            sum1 += term
            term *= (arrival_rate / service_rate) / (n + 1)
        lq = (term * rho) / (math.pow((1 - rho), 2)) * (1 / (sum1 + (term / (1 - rho))))
        return (lq / arrival_rate) + (1 / service_rate)
    p = 1
    q = 1
    exp_n_system = 0
    for n in range(1, servers + 1):
        p = arrival_rate / (n * service_rate) * p
        q += p
        exp_n_system = exp_n_system + n * p
    for n in range(servers + 1, servers + queue_capacity + 1):
        p = arrival_rate / (service_rate * servers) * p
        q += p
        exp_n_system = exp_n_system + n * p
    lambda_bar = arrival_rate * (1 - p / q)
    return (exp_n_system / q) / lambda_bar


def PrWait(arrival_rate, service_rate, servers, queue_capacity=None):
    """
    :param arrival_rate:
    :param service_rate:
    :param servers:
    :param queue_capacity:
    :return:
    """
    p = 1
    q = 1
    if queue_capacity is None:
        rho = arrival_rate / (service_rate * servers)
        if rho > 1:
            raise ValueError(UtilMsg)
        if servers == 1:
            return 1 - (1 / (1 + (((arrival_rate / service_rate) / (servers + 1)) / (1 - rho))))
        for n in range(1, servers + 1):
            p = arrival_rate / (n * service_rate) * p
            q += p
            if n == servers - 1 or servers == 1:
                Qtemp = q
        print(Qtemp, q, p, rho)
        return 1 - Qtemp / (q + p * rho / (1 - rho))
    if servers == 1:
        p = arrival_rate / service_rate * p
        q += p
        for n in (servers + 1, servers + queue_capacity + 1):
            p = arrival_rate / (servers * service_rate) * p
            q += p
        print("q", q, p)
        return 1 - (1 / q)
    for n in range(1, servers + 1):
        p = arrival_rate / (n * service_rate) * p
        q += p
        if n == servers - 1 or servers == 1:
            Qtemp = q
    for n in range(servers + 1, servers + queue_capacity + 1):
        p = arrival_rate / (service_rate * servers) * p
        q += p
    return 1 - Qtemp / q


def PrBalk(arrival_rate, service_rate, servers, queue_capacity=None):
    """
    :param arrival_rate:
    :param service_rate:
    :param servers:
    :param queue_capacity:
    :return:
    """
    p = 1
    q = 1
    if queue_capacity is None:
        return 0
    for n in range(1, servers + 1):
        p = arrival_rate / (n * service_rate) * p
        q += p
    for n in range(servers + 1, servers + queue_capacity + 1):
        p = arrival_rate / (service_rate * servers) * p
        q += p
    return p / q


def error_less_prwait(arrival_rate, service_rate, servers):
    p = 1
    q = 1
    rho = arrival_rate / (servers * service_rate)
    if servers == 1:
        return rho
    for n in range(1, servers + 1):
        p = arrival_rate / (n * service_rate) * p
        Qtemp = q
        q = q + p
    if rho >= 1:
        return 0.99999999
    else:
        return 1 - Qtemp / (q + p * rho / (1 - rho))


def service_level(threshold_time, arrival_rate, service_rate, servers, queue_capacity=None):
    if queue_capacity is None:
        p = arrival_rate / (servers * service_rate)
        temp1 = error_less_prwait(arrival_rate, service_rate, servers)
        temp2 = math.exp(-servers * service_rate * (1 - p) * threshold_time)
        return 1 - (temp1 * temp2)
    r = 1
    sum1 = r
    for i in range(1, servers):
        r = (arrival_rate / (i * service_rate)) * r
        sum1 = sum1 + r
    a = servers * service_rate * threshold_time
    p = arrival_rate / (servers * service_rate)
    q = math.exp(-a)
    u = q
    sum2 = 0
    for i in range(1, queue_capacity + 1):
        r = p * r
        sum2 = sum2 + r * u
        sum1 = sum1 + r
        q = (a / i) * q
        u = u + q
    return 1 - sum2 / sum1


def min_agents(threshold_time, exp_service_level, arrival_rate, service_rate, system_capacity=None):
    exp_service_level = 1 - exp_service_level
    if system_capacity is None:
        servers = int((arrival_rate / service_rate) + 0.5)
        if servers == 0: servers = 1
        found_min = False
        while not found_min:
            rho = arrival_rate / (servers * service_rate)
            temp1 = error_less_prwait(arrival_rate, service_rate, servers)
            temp2 = math.exp(-servers * service_rate * (1 - rho) * threshold_time)
            if temp1 * temp2 < exp_service_level:
                found_min = True
            else:
                servers += 1
        return servers
    servers = 1
    found_min = False
    while not found_min and servers <= system_capacity:
        temp1 = 1 - service_level(threshold_time, arrival_rate, service_rate, servers, (system_capacity - servers))
        if temp1 < exp_service_level:
            found_min = True
        else:
            servers += 1
    if servers > system_capacity:
        return -1
    else:
        return servers


def min_servers(threshold_time, exp_service_level, arrival_rate, service_rate, queue_capacity=None):
    exp_service_level = 1 - exp_service_level
    if queue_capacity is None:
        servers = int((arrival_rate / service_rate) + 0.5)
        if servers == 0: servers = 1
        found_min = False
        while not found_min:
            rho = arrival_rate / (servers * service_rate)
            temp1 = error_less_prwait(arrival_rate, service_rate, servers)
            temp2 = math.exp(-servers * service_rate * (1 - rho) * threshold_time)
            if temp1 * temp2 < exp_service_level:
                found_min = True
            else:
                servers += 1
        return servers
    servers = 1
    found_min = False
    while not found_min:
        temp1 = 1 - service_level(threshold_time, arrival_rate, service_rate, servers, queue_capacity)
        if temp1 < exp_service_level:
            found_min = True
        else:
            servers += 1
    return servers


if __name__ == "__main__":
    a, b, c, d = 100, 0.5, 2, 3
    t, level = 0.1, 0.9
    print(util(a, b, c, d))
    print(Po(a, b, c, d))
    print(Lq(a, b, c, d))
    print(L(a, b, c, d))
    print(Wq(a, b, c, d))
    print(W(a, b, c, d))
    print(PrWait(a, b, c, d))
    print(PrBalk(a, b, c, d))
    print(service_level(t, a, b, c, d))
    print(min_servers(t, level, a, b, d))
    print(min_agents(t, level, a, b, c + d))

    # [4.950000000249975e-11, 2.98989900015099, 4.98989899015199, 4.98989901509899, 2.98989901509899, 0.9999999950005, 0.9999999900505, 0.990000000049995, 0.00020984310382465843, 93, 5]


