

8核16G内存

 - 峰值占用：16*(0.53-0.07) = 7.36 GB


```
   Ordered by: cumulative time
   List reduced from 529 to 5 due to restriction <5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1   14.695   14.695   70.686   70.686 /workspace/cs336_basics/bpe_train_copy.py:94(train_bpe)
        9    0.000    0.000   43.872    4.875 /usr/lib/python3.12/multiprocessing/process.py:142(join)
        9    0.000    0.000   43.872    4.875 /usr/lib/python3.12/multiprocessing/popen_fork.py:36(wait)
       47    0.000    0.000   43.870    0.933 /usr/lib/python3.12/multiprocessing/popen_fork.py:24(poll)
       46   43.870    0.954   43.870    0.954 {built-in method posix.waitpid}
       41    8.520    0.208    8.520    0.208 {method 'read' of '_io.BufferedReader' objects}
       18    1.642    0.091    1.642    0.091 {method 'decode' of 'bytes' objects}
        1    0.000    0.000    1.294    1.294 /workspace/cs336_basics/bpe_train_copy.py:9(find_chunk_boundaries)
  1456962    0.745    0.000    0.859    0.000 /usr/lib/python3.12/collections/__init__.py:595(__init__)
  1351526    0.361    0.000    0.361    0.000 /usr/lib/python3.12/collections/__init__.py:737(__delitem__)



➜  /workspace git:(main) ✗ ls -l -h data
total 2.1G
-rw-r--r-- 1 root root 2.1G Feb 27 21:05 TinyStoriesV2-GPT4-train.txt
-rw-r--r-- 1 root root 108K Feb 27 21:17 train_bpe_merges.pkl
-rw-r--r-- 1 root root 133K Feb 27 21:17 train_bpe_merges.txt
-rw-r--r-- 1 root root 115K Feb 27 21:17 train_bpe_vocab.pkl
-rw-r--r-- 1 root root 145K Feb 27 21:17 train_bpe_vocab.txt
```


```
➜  data git:(main) ✗ head train_bpe_vocab.txt 
0       b'<|endoftext|>'
1       b'\x00'
2       b'\x01'
3       b'\x02'
4       b'\x03'
5       b'\x04'
6       b'\x05'
7       b'\x06'
8       b'\x07'
9       b'\x08'

➜  data git:(main) ✗ tail train_bpe_vocab.txt 
9990    b' whiskers'
9991    b' nicest'
9992    b' improving'
9993    b' booth'
9994    b' Land'
9995    b'Surrender'
9996    b'Rocky'
9997    b' meadows'
9998    b' imaginary'
9999    b' bold'
```


```
➜  data git:(main) ✗ head train_bpe_merges.txt 
b' ' b't'
b'h' b'e'
b' ' b'a'
b' ' b's'
b' ' b'w'
b'n' b'd'
b' t' b'he'
b'e' b'd'
b' ' b'b'
b' t' b'o'

➜  data git:(main) ✗ tail  train_bpe_merges.txt
b' wh' b'iskers'
b' nice' b'st'
b' impro' b'ving'
b' bo' b'oth'
b' L' b'and'
b'S' b'urrender'
b'Rock' b'y'
b' meadow' b's'
b' imag' b'inary'
b' bo' b'ld'
```