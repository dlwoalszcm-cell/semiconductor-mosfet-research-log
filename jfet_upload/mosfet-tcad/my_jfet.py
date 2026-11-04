'''해야할 것.
1. JFET 하나 생성
2. Drain 0.05 V
3. Vg 0 V ~ -2 V까지 0.5 V씩 내리기
4. Id와 열린 채널 두께 재기
5. 결과 출력
'''
from jfet import Jfet

dev = Jfet(L_um=5, a_um=0.6, ND=1e16)
dev.bias(drain=0.05)

for vg in [0.0, -0.5, -1.0, -1.5, -2.0]:
    dev.bias(gate=vg)
    i = dev.current()
    print(f"vg = {vg} V, Id = {i*1e6:.3f} uA/um")
    print(f"Channel opening = {dev.opening():.3f} um")
