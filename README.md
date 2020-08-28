# DMDmethod

**使い方**
1. csvファイルにinp
2. main.pyで実行

***
**(メモ)**
<br>
1. pyx(Cythonコードファイル)のコンパイル  

 **setup.py**
 ```
 from distutils.core import setup
 from distutils.extension import Extension
 from Cython.Distutils import build_ext

 import numpy as np

 sourcefiles = ['hogehoge.pyx']
 setup(
     cmdclass = {'build_ext':build_ext},
     ext_modules = [Extension('piyopiyo', sourcefiles)],
     include_dirs = [np.get_include()]
 )
 ```

 以下のコマンドの実行
 `python setup.py build_ext --inplace`  
<br>

2. 実行方法

**main.py**
```
from piyopiyo import func1 #とか
import piyopiyo #とかすることで

if __name__ == "__main__":
  piyopiyo.func2(x)

```

***
ここら辺を参考にする
> https://qiita.com/en3/items/1f1a609c4d7c8f3066a7
> https://medium.com/lsc-psd/cython%E5%B0%8E%E5%85%A5%E3%81%AE%E3%81%84%E3%82%8D%E3%81%AF-ca8f93f804f4
