import pywt
from matplotlib import pyplot as plt


def swt_decom(data, wavefunc, lv):

    if lv == 1:
        [(cA1,cD1)] = pywt.swt(data,wavefunc,level=lv,axis=0)

        coffes = [cA1, cD1]
        return coffes
        #return cA1,cD1
    elif lv == 2:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2,cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)

        coffes = [cA2,cD2,cD1]
        return coffes
        #return cA2,cD2,cD1

    elif lv == 3:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2,cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3,cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)

        coffes = [cA3,cD3,cD2,cD1]
        return coffes
        #return cA3,cD3,cD2,cD1

    elif lv == 4:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)

        coffes = [cA4, cD4, cD3, cD2, cD1]
        return coffes
        #return cA4, cD4, cD3, cD2, cD1

    elif lv == 5:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)

        coffes = [cA5, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes
        #return [(np.asarray(cA), np.asarray(cD)) for cA, cD in ret]

    elif lv == 6:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        coffes = [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 7:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        coffes = [cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 8:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        coffes = [cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 9:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1, wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2, wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        [(cA9, cD9)] = pywt.swt(cA8, wavefunc, level=1, axis=0)
        coffes = [cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 10:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        [(cA9, cD9)] = pywt.swt(cA8, wavefunc, level=1, axis=0)
        [(cA10, cD10)] = pywt.swt(cA9, wavefunc, level=1, axis=0)
        coffes = [cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 11:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        [(cA9, cD9)] = pywt.swt(cA8, wavefunc, level=1, axis=0)
        [(cA10, cD10)] = pywt.swt(cA9, wavefunc, level=1, axis=0)
        [(cA11, cD11)] = pywt.swt(cA10, wavefunc, level=1, axis=0)
        coffes = [cA11, cD11, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 12:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        [(cA9, cD9)] = pywt.swt(cA8, wavefunc, level=1, axis=0)
        [(cA10, cD10)] = pywt.swt(cA9, wavefunc, level=1, axis=0)
        [(cA11, cD11)] = pywt.swt(cA10, wavefunc, level=1, axis=0)
        [(cA12, cD12)] = pywt.swt(cA11, wavefunc, level=1, axis=0)
        coffes = [cA12, cD12, cD11, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 13:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        [(cA9, cD9)] = pywt.swt(cA8, wavefunc, level=1, axis=0)
        [(cA10, cD10)] = pywt.swt(cA9, wavefunc, level=1, axis=0)
        [(cA11, cD11)] = pywt.swt(cA10, wavefunc, level=1, axis=0)
        [(cA12, cD12)] = pywt.swt(cA11, wavefunc, level=1, axis=0)
        [(cA13, cD13)] = pywt.swt(cA12, wavefunc, level=1, axis=0)
        coffes = [cA13, cD13, cD12, cD11, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 14:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        [(cA9, cD9)] = pywt.swt(cA8, wavefunc, level=1, axis=0)
        [(cA10, cD10)] = pywt.swt(cA9, wavefunc, level=1, axis=0)
        [(cA11, cD11)] = pywt.swt(cA10, wavefunc, level=1, axis=0)
        [(cA12, cD12)] = pywt.swt(cA11, wavefunc, level=1, axis=0)
        [(cA13, cD13)] = pywt.swt(cA12, wavefunc, level=1, axis=0)
        [(cA14, cD14)] = pywt.swt(cA13, wavefunc, level=1, axis=0)
        coffes = [cA14, cD14, cD13, cD12, cD11, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 15:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        [(cA9, cD9)] = pywt.swt(cA8, wavefunc, level=1, axis=0)
        [(cA10, cD10)] = pywt.swt(cA9, wavefunc, level=1, axis=0)
        [(cA11, cD11)] = pywt.swt(cA10, wavefunc, level=1, axis=0)
        [(cA12, cD12)] = pywt.swt(cA11, wavefunc, level=1, axis=0)
        [(cA13, cD13)] = pywt.swt(cA12, wavefunc, level=1, axis=0)
        [(cA14, cD14)] = pywt.swt(cA13, wavefunc, level=1, axis=0)
        [(cA15, cD15)] = pywt.swt(cA14, wavefunc, level=1, axis=0)
        coffes = [cA15, cD15, cD14, cD13, cD12, cD11, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 16:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        [(cA9, cD9)] = pywt.swt(cA8, wavefunc, level=1, axis=0)
        [(cA10, cD10)] = pywt.swt(cA9, wavefunc, level=1, axis=0)
        [(cA11, cD11)] = pywt.swt(cA10, wavefunc, level=1, axis=0)
        [(cA12, cD12)] = pywt.swt(cA11, wavefunc, level=1, axis=0)
        [(cA13, cD13)] = pywt.swt(cA12, wavefunc, level=1, axis=0)
        [(cA14, cD14)] = pywt.swt(cA13, wavefunc, level=1, axis=0)
        [(cA15, cD15)] = pywt.swt(cA14, wavefunc, level=1, axis=0)
        [(cA16, cD16)] = pywt.swt(cA15, wavefunc, level=1, axis=0)
        coffes = [cA16, cD16, cD15, cD14, cD13, cD12, cD11, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 17:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        [(cA9, cD9)] = pywt.swt(cA8, wavefunc, level=1, axis=0)
        [(cA10, cD10)] = pywt.swt(cA9, wavefunc, level=1, axis=0)
        [(cA11, cD11)] = pywt.swt(cA10, wavefunc, level=1, axis=0)
        [(cA12, cD12)] = pywt.swt(cA11, wavefunc, level=1, axis=0)
        [(cA13, cD13)] = pywt.swt(cA12, wavefunc, level=1, axis=0)
        [(cA14, cD14)] = pywt.swt(cA13, wavefunc, level=1, axis=0)
        [(cA15, cD15)] = pywt.swt(cA14, wavefunc, level=1, axis=0)
        [(cA16, cD16)] = pywt.swt(cA15, wavefunc, level=1, axis=0)
        [(cA17, cD17)] = pywt.swt(cA16, wavefunc, level=1, axis=0)
        coffes = [cA17, cD17, cD16, cD15, cD14, cD13, cD12, cD11, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes

    elif lv == 18:
        [(cA1, cD1)] = pywt.swt(data, wavefunc, level=1,axis=0)
        [(cA2, cD2)] = pywt.swt(cA1,wavefunc,level=1,axis=0)
        [(cA3, cD3)] = pywt.swt(cA2,wavefunc,level=1,axis=0)
        [(cA4, cD4)] = pywt.swt(cA3, wavefunc, level=1,axis=0)
        [(cA5, cD5)] = pywt.swt(cA4, wavefunc, level=1,axis=0)
        [(cA6, cD6)] = pywt.swt(cA5, wavefunc, level=1, axis=0)
        [(cA7, cD7)] = pywt.swt(cA6, wavefunc, level=1, axis=0)
        [(cA8, cD8)] = pywt.swt(cA7, wavefunc, level=1, axis=0)
        [(cA9, cD9)] = pywt.swt(cA8, wavefunc, level=1, axis=0)
        [(cA10, cD10)] = pywt.swt(cA9, wavefunc, level=1, axis=0)
        [(cA11, cD11)] = pywt.swt(cA10, wavefunc, level=1, axis=0)
        [(cA12, cD12)] = pywt.swt(cA11, wavefunc, level=1, axis=0)
        [(cA13, cD13)] = pywt.swt(cA12, wavefunc, level=1, axis=0)
        [(cA14, cD14)] = pywt.swt(cA13, wavefunc, level=1, axis=0)
        [(cA15, cD15)] = pywt.swt(cA14, wavefunc, level=1, axis=0)
        [(cA16, cD16)] = pywt.swt(cA15, wavefunc, level=1, axis=0)
        [(cA17, cD17)] = pywt.swt(cA16, wavefunc, level=1, axis=0)
        [(cA18, cD18)] = pywt.swt(cA17, wavefunc, level=1, axis=0)
        coffes = [cA18, cD18, cD17, cD16, cD15, cD14, cD13, cD12, cD11, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
        #return cA5, cD5, cD4, cD3, cD2, cD1
        return coffes


def iswt_decom(data, wavefunc):

    lv = len(data) - 1
    if lv == 1:
        y = pywt.iswt([(data[0],data[1])],wavefunc)
        #[(cA1,cD1)] = pywt.swt(data,wavefunc,level=lv)
        return y

    elif lv == 2:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[-1])],wavefunc)

        return y2

    elif lv == 3:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)

        return y3

    elif lv == 4:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)

        return y4

    elif lv == 5:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[3])], wavefunc)
        y5 = pywt.iswt([(y4, data[3])], wavefunc)

        return y5

    elif lv == 6:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)

        return y6

    elif lv == 7:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)

        return y7

    elif lv == 8:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)

        return y8

    elif lv == 9:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)
        y9 = pywt.iswt([(y8, data[9])], wavefunc)

        return y9

    elif lv == 10:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)
        y9 = pywt.iswt([(y8, data[9])], wavefunc)
        y10 = pywt.iswt([(y9, data[10])], wavefunc)

        return y10

    elif lv == 11:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)
        y9 = pywt.iswt([(y8, data[9])], wavefunc)
        y10 = pywt.iswt([(y9, data[10])], wavefunc)
        y11 = pywt.iswt([(y10, data[11])], wavefunc)

        return y11

    elif lv == 12:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)
        y9 = pywt.iswt([(y8, data[9])], wavefunc)
        y10 = pywt.iswt([(y9, data[10])], wavefunc)
        y11 = pywt.iswt([(y10, data[11])], wavefunc)
        y12 = pywt.iswt([(y11, data[12])], wavefunc)

        return y12
    elif lv == 13:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)
        y9 = pywt.iswt([(y8, data[9])], wavefunc)
        y10 = pywt.iswt([(y9, data[10])], wavefunc)
        y11 = pywt.iswt([(y10, data[11])], wavefunc)
        y12 = pywt.iswt([(y11, data[12])], wavefunc)
        y13 = pywt.iswt([(y12, data[13])], wavefunc)

        return y13

    elif lv == 14:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)
        y9 = pywt.iswt([(y8, data[9])], wavefunc)
        y10 = pywt.iswt([(y9, data[10])], wavefunc)
        y11 = pywt.iswt([(y10, data[11])], wavefunc)
        y12 = pywt.iswt([(y11, data[12])], wavefunc)
        y13 = pywt.iswt([(y12, data[13])], wavefunc)
        y14 = pywt.iswt([(y13, data[14])], wavefunc)

        return y14

    elif lv == 15:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)
        y9 = pywt.iswt([(y8, data[9])], wavefunc)
        y10 = pywt.iswt([(y9, data[10])], wavefunc)
        y11 = pywt.iswt([(y10, data[11])], wavefunc)
        y12 = pywt.iswt([(y11, data[12])], wavefunc)
        y13 = pywt.iswt([(y12, data[13])], wavefunc)
        y14 = pywt.iswt([(y13, data[14])], wavefunc)
        y15 = pywt.iswt([(y14, data[15])], wavefunc)

        return y15

    elif lv == 16:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)
        y9 = pywt.iswt([(y8, data[9])], wavefunc)
        y10 = pywt.iswt([(y9, data[10])], wavefunc)
        y11 = pywt.iswt([(y10, data[11])], wavefunc)
        y12 = pywt.iswt([(y11, data[12])], wavefunc)
        y13 = pywt.iswt([(y12, data[13])], wavefunc)
        y14 = pywt.iswt([(y13, data[14])], wavefunc)
        y15 = pywt.iswt([(y14, data[15])], wavefunc)
        y16 = pywt.iswt([(y15, data[16])], wavefunc)

        return y16

    elif lv == 17:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)
        y9 = pywt.iswt([(y8, data[9])], wavefunc)
        y10 = pywt.iswt([(y9, data[10])], wavefunc)
        y11 = pywt.iswt([(y10, data[11])], wavefunc)
        y12 = pywt.iswt([(y11, data[12])], wavefunc)
        y13 = pywt.iswt([(y12, data[13])], wavefunc)
        y14 = pywt.iswt([(y13, data[14])], wavefunc)
        y15 = pywt.iswt([(y14, data[15])], wavefunc)
        y16 = pywt.iswt([(y15, data[16])], wavefunc)
        y17 = pywt.iswt([(y16, data[17])], wavefunc)

        return y17

    elif lv == 18:
        y1 = pywt.iswt([(data[0],data[1])],wavefunc)
        y2 = pywt.iswt([(y1,data[2])],wavefunc)
        y3 = pywt.iswt([(y2, data[3])], wavefunc)
        y4 = pywt.iswt([(y3, data[4])], wavefunc)
        y5 = pywt.iswt([(y4, data[5])], wavefunc)
        y6 = pywt.iswt([(y5, data[6])], wavefunc)
        y7 = pywt.iswt([(y6, data[7])], wavefunc)
        y8 = pywt.iswt([(y7, data[8])], wavefunc)
        y9 = pywt.iswt([(y8, data[9])], wavefunc)
        y10 = pywt.iswt([(y9, data[10])], wavefunc)
        y11 = pywt.iswt([(y10, data[11])], wavefunc)
        y12 = pywt.iswt([(y11, data[12])], wavefunc)
        y13 = pywt.iswt([(y12, data[13])], wavefunc)
        y14 = pywt.iswt([(y13, data[14])], wavefunc)
        y15 = pywt.iswt([(y14, data[15])], wavefunc)
        y16 = pywt.iswt([(y15, data[16])], wavefunc)
        y17 = pywt.iswt([(y16, data[17])], wavefunc)
        y18 = pywt.iswt([(y17, data[18])], wavefunc)

        return y18


def main():

    #load data
    #file_name = ''

    x = [1, 2, 3, 4, 6, 7, 8, 19, 22, 1, 23, 40]

    wavefun = 'db1'

    #cA1, cD1 = swt_decom(x, wavefun, 1)
    #data = [cA1,cD1]
    # cA2,cD2,cD1 = swt_decom(x,wavefun,2)
    # data = [cA2,cD2,cD1]
    data = swt_decom(x, wavefun, 10)

    y = iswt_decom(data,wavefun)
    ## figure

    plt.subplot(5,1,1),plt.plot(x,'k'),plt.title('orignal')
    plt.subplot(5, 1, 2), plt.plot(data[0],'r'), plt.title('cA10')
    plt.subplot(5, 1, 3), plt.plot(data[1]), plt.title('cD2')
    plt.subplot(5, 1, 4), plt.plot(data[2]), plt.title('cD1')

    # plt.subplot(4, 1, 2), plt.plot(cA1), plt.title('cA1')
    # plt.subplot(4, 1, 3), plt.plot(cD1), plt.title('cD1')
    plt.subplot(5, 1, 5), plt.plot(y), plt.title('recom')


    plt.show()




if __name__ == '__main__':
    main()



