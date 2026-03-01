



# 【1】

显存占用：
  - 30GB/45GB
  - GPU：68%

```
(cs336-basics) ➜  /workspace git:(main) ✗ nvidia-smi 
Sun Mar  1 19:52:04 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L40                     On  |   00000000:23:00.0 Off |                    0 |
| N/A   51C    P0            269W /  300W |   31687MiB /  46068MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```



```
Step 297/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.57 | eval:  inf
Step 298/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.51 | eval:  inf
Step 299/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.53 | eval:  inf
Step 300/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.51 | eval:  inf
prompt:  Once upon a time
Answer:  , there was a little girl named Tim. Tim liked to play with his friends, and Tim would have a good friend.
Lily
One day, Max's mom saw a big, there was a big car. Tim jumped in the ball. Tim was very happy. He had a big, time, a little girl named Sam. Tim was sad, but she walked to the box. Tim was very excited because Max was sad. Sam was still happy to go to the stairs together. He was very happy.
The dog saw a little girl in the garden. She wanted to play into a big, hat. The book so happy. The cat saw the tree and said, "I too new!" She said, "I have a high with the ball and a big, friend,."
The little boy was very happy. She wanted to do the big house. Tim thought out, "Yes, I am sorry for the funny." It was a big, colorful little girl namedOnce upon a little girl named Lily. Sue was very happy and would play with her friends.
At the park, Lily and his mom played with the fence. They had there what was it was. They saw a green big bird with her mom. In
Gen-length:  971 , Token Count:  252
Step 300/8258 - Sample: 1.09s | 
Step 301/8258 - Total: 0.80s | Forward: 0.00s |Backward: 0.01s | Optim: 0.00s |Loss:  3.56 | eval:  inf
Step 302/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.48 | eval:  inf
Step 303/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.54 | eval:  inf
Step 304/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.55 | eval:  inf
Step 305/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.47 | eval:  inf
Step 306/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.50 | eval:  inf
Step 307/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.51 | eval:  inf
Step 308/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.50 | eval:  inf
Step 309/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.48 | eval:  inf
Step 310/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  3.44 | eval:  inf
```



# 【2】


Step 2896/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.66 | eval:  1.75
Step 2897/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.66 | eval:  1.75
Step 2898/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.66 | eval:  1.75
Step 2899/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2900/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
prompt:  Once upon a time
Answer:  , there was a little girl named Lily. She was a very independent girl. One day, Lily and her mom went to the park.
At the park, Lily saw a big tree. She wanted to climb it. She said, "Mom, can I climb the tree?" Her mom smiled and said, "No, Lily. We will not climb the tree."
Lily climbed the tree and found a big, red ball. She was so happy! Lily played with the ball all day. She knew that climbing the tree could be fun, but only under the tree.

Gen-length:  450 , Token Count:  118
Step 2900/8258 - Sample: 0.52s | 
Step 2901/8258 - Total: 0.79s | Forward: 0.00s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2902/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2903/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2904/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2905/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2906/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2907/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2908/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2909/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2910/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2911/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2912/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2913/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2914/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2915/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2916/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2917/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2918/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2919/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2920/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2921/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2922/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2923/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2924/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2925/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2926/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.67 | eval:  1.75
Step 2927/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2928/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2929/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2930/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2931/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.66 | eval:  1.75
Step 2932/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2933/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2934/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2935/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.67 | eval:  1.75
Step 2936/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2937/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2938/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2939/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2940/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2941/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2942/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2943/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2944/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2945/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.66 | eval:  1.75
Step 2946/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2947/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2948/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2949/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2950/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2951/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2952/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2953/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2954/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2955/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2956/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.59 | eval:  1.75
Step 2957/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2958/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2959/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2960/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.57 | eval:  1.75
Step 2961/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2962/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2963/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2964/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.66 | eval:  1.75
Step 2965/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2966/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2967/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2968/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2969/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2970/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.57 | eval:  1.75
Step 2971/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2972/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2973/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2974/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2975/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2976/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2977/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2978/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2979/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2980/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2981/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.59 | eval:  1.75
Step 2982/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2983/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2984/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2985/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2986/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2987/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2988/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2989/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Step 2990/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2991/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.75
Step 2992/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2993/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2994/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.75
Step 2995/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.75
Step 2996/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2997/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.75
Step 2998/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.75
Step 2999/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.75
Eval steps: 1000, one stpe len: 65536, all token len: 5465882
New best eval loss: 1.6277
Step 3000/8258 - Eval and save checkpoint: 267.33s | 
Step 3000/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.63
prompt:  Once upon a time
Answer:  , there was a big, wide tree. It had a lot of pretty colors. Many animals lived in the tree. They loved to play and have fun.
One day, a little bird came to the tree. The bird was sad. The bird said, "Why are you sad, tree?" The tree said, "I can't find my own home."
The tree wanted to help the bird. It had an idea. It said, "I will help you find your home. We can be friends." The tree was happy.
The tree and the bird looked for the bird's home. They found many things to do. They found things that were right. But the bird told the tree to be happy and not ashamed anymore. The tree, the bird, and the tree became good friends.

Gen-length:  633 , Token Count:  165
Step 3000/8258 - Sample: 0.71s | 
Step 3001/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.63
Step 3002/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.63
Step 3003/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.63
Step 3004/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.63
Step 3005/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.60 | eval:  1.63
Step 3006/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.63
Step 3007/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.63
Step 3008/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.62 | eval:  1.63
Step 3009/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.63
Step 3010/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.63
Step 3011/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.59 | eval:  1.63
Step 3012/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.63
Step 3013/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.64 | eval:  1.63
Step 3014/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.65 | eval:  1.63
Step 3015/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.63
Step 3016/8258 - Total: 0.79s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.63 | eval:  1.63
Step 3017/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.67 | eval:  1.63
Step 3018/8258 - Total: 0.80s | Forward: 0.01s |Backward: 0.01s | Optim: 0.00s |Loss:  1.61 | eval:  1.63
```