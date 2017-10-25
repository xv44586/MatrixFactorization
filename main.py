"""
based on matrix factorization to predict the blank values in the matrix.
"""


from numpy import *


class Factorization(object):
    def __init__(self, path, rank):
        self.path = path
        self.rank = rank

    def load_data(self):
        with open(self.path) as f:
            data = []
            for line in f.readlines():
                arr = []
                lines = line.strip().split()
                for x in lines:
                    if x.isdigit():
                        arr.append(float(x))
                    else:
                        arr.append(float(0))
                # print(arr)
                data.append(arr)
        # print data
        return data

    @staticmethod
    def gradAscent(data, K, batch=10000):
        dataMat = mat(data)
        print('dataMat:\n', dataMat)
        m, n = shape(dataMat)
        p = mat(random.random((m, K)))
        q = mat(random.random((K, n)))

        alpha = 0.0002
        beta = 0.02
        maxCycles = batch

        for step in range(maxCycles):
            for i in range(m):
                for j in range(n):
                    if dataMat[i, j] > 0:
                        # print dataMat[i,j]
                        error = dataMat[i, j]
                        for k in range(K):
                            error = error - p[i, k] * q[k, j]
                        for k in range(K):
                            p[i, k] = p[i, k] + alpha * (2 * error * q[k, j] - beta * p[i, k])
                            q[k, j] = q[k, j] + alpha * (2 * error * p[i, k] - beta * q[k, j])

            loss = 0.0
            for i in range(m):
                for j in range(n):
                    if dataMat[i, j] > 0:
                        error = 0.0
                        for k in range(K):
                            error = error + p[i, k] * q[k, j]
                        loss = (dataMat[i, j] - error) * (dataMat[i, j] - error)
                        for k in range(K):
                            loss = loss + beta * (p[i, k] * p[i, k] + q[k, j] * q[k, j]) / 2

            if loss < 0.001:
                break
            # print step
            if step % 1000 == 0:
                print('loss: ', loss)

        return p, q

    def __call__(self, *args, **kwargs):
        data = self.load_data()
        p, q = self.gradAscent(data, self.rank)
        return p * q


if __name__ == "__main__":
    f = Factorization("./data", 5)

    result = f()

    print('result: \n', result)
