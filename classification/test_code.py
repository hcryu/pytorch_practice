import torch


"""
Test View

x = torch.randn(2, 3)
print("x", x)
#y = x.view(3, 2)
y = x.reshape(3, 2)

print("y", y)
y[0][0] = 100
print("After: x", x)
print("After: y", y)
#z = x.t()  # 텐서 전치(Transpose) -> 메모리 비연속
#print(z.is_contiguous())  # False

#y = z.view(3, 2)  # ❌ RuntimeError 발생

"""

import time

#for i in range(10):
    #time.sleep(0.5)  # 0.5초 대기 (예제용)
    #print(f"Processing {i+1}/10")
    

from tqdm import tqdm
import time

for i in tqdm(range(10)):  # tqdm()으로 감싸주기
    time.sleep(0.5)  # 0.5초 대기

