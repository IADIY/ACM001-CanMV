# ACM001 Quick Start Tutorial

## Run Your First Example on ACM001


## Run Your First AI Model on ACM001


## UART Commnuication
```
from machine import UART
from machine import FPIOA
import time

fpioa = FPIOA()

# UART2 Initialize
fpioa.set_function(44,FPIOA.UART2_TXD)
fpioa.set_function(45,FPIOA.UART2_RXD)

uart=UART(UART.UART2,115200) #Configure Baudrate


while True:
    text = uart.read(32)
    if text:
        print(text)
    uart.write('Testing')
    time.sleep(0.1) #100ms

```
