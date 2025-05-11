import torch
import time

def cuda_available():
    print(f'cuda available = {torch.cuda.is_available()}')


def benchmark():
    matrix_size = 32*256
    x = torch.randn(matrix_size,matrix_size)
    y = torch.randn(matrix_size,matrix_size)

    print('--------------- CPU BENCHMARK--------------------')
    start = time.time()
    res = torch.matmul(x,y)
    print(time.time() - start)
    print(res.device)

    device = torch.device('cuda')
    x_gpu = x.to(device)
    y_gpu = x.to(device)
    torch.cuda.synchronize()

    for i in range(5):
        print('--------------- GPU BENCHMARK--------------------')
        start = time.time()
        res = torch.matmul(x_gpu,y_gpu)
        torch.cuda.synchronize()
        print(time.time() - start)
        print(res.device)



if __name__ == '__main__':
    cuda_available()
    benchmark()
