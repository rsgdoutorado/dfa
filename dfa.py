import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def dfa_monofractal(signal, m, scale):
    x = np.cumsum(signal - np.mean(signal))
    x = np.transpose(x)
    segments = []
    f = []
    for ns in range(len(scale)):
        segments.append((int)(np.floor(len(x) / scale[ns])))
        rms = []
        for v in range(1, segments[ns] + 1):
            idx_start = ((v - 1) * scale[ns])
            idx_stop = v * scale[ns]
            index = np.arange(idx_start, idx_stop)
            x_idx = x[index[0]:index[-1] + 1]
            c = np.polyfit(index, x_idx, m)
            fit = np.polyval(c, index)
            rms.append(np.sqrt(np.mean(np.power(x_idx - fit, 2))))
        f.append(np.sqrt(np.mean(np.power(rms, 2))))
    return f


def dfa_multifractal(signal, m, scale, q):
    x = np.cumsum(signal - np.mean(signal))
    x = np.transpose(x)
    segments = []
    Fq = np.zeros((len(q), len(scale)))
    for ns in range(len(scale)):
        segments.append((int)(np.floor(len(x) / scale[ns])))
        rms = []
        for v in range(1, segments[ns] + 1):
            idx_start = ((v - 1) * scale[ns])
            idx_stop = v * scale[ns]
            index = np.arange(idx_start, idx_stop)
            x_idx = x[index[0]:index[-1] + 1]
            c = np.polyfit(index, x_idx, m)
            fit = np.polyval(c, index)
            rms.append(np.sqrt(np.mean(np.power(x_idx - fit, 2))))
        qrms = []
        fq = []
        for nq in range(len(q)):
            qrms.append(np.power(rms, q[nq]))
            fq.append(np.power(np.mean(qrms[nq]), (1/q[nq])))
        fq_array = np.array(fq)
        fq_array[q == 0] = np.exp(0.5 * np.mean(np.log(np.power(rms, 2))))
        Fq[:, ns] = fq_array
    Hq = np.zeros((1, len(q)))
    qRegLine = np.zeros((len(q), len(scale)))
    for nq in range(len(q)):
        c = np.polyfit(np.log2(scale), np.log2(Fq[nq, :]), 1)
        Hq[0, nq] = c[0]
        qRegLine[nq] = np.polyval(c, np.log2(scale))
    tq = Hq*q-1
    hq = np.diff(tq)/(q[1]-q[0])
    q1 = np.transpose(q[0:len(q)-1])
    Dq = (q1*hq)-tq[0, 0:tq.size-1]
    return Fq, Hq, qRegLine, tq, hq, Dq

