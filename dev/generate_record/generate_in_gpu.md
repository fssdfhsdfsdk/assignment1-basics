


```
(cs336-basics) ➜  cs336_basics git:(main) ✗ ls -l -h checkpoints/my_first_llm 
total 520M
-rw-r--r-- 1 root root    0 Mar  1 20:40 eval_loss_1.6277340097427369_step_3000.pt
-rw-r--r-- 1 root root 260M Mar  1 20:22 eval_loss_1.7520343259572984_step_2000.pt
-rw-r--r-- 1 root root 260M Mar  1 20:04 eval_loss_2.1214684953689575_step_1000.pt
```


```
(cs336-basics) ➜  cs336_basics git:(main) ✗ python generate.py
prompt:  Once upon a time
Answer:  , there was a big, friendly dog named Max. Max loved to play and run in the park. One day, Max saw a little bird with a hurt wing. Max wanted to help the bird, so he went to ask his friend, the wise old owl, for help.
"Mr. Owl, can you help me?" asked the owl. The owl looked at Max and said, "I am sad because I lost my family. I am sad."
Max wanted to help the owl. He and the owl went to find the family. They looked high and low. Finally, they found the owl's family. The owl was so happy! They all became good friends.
The owl told Max that in the forest was a happy place for all his family. Max felt happy too. The moral of the story is that friends help each other and being kind to others.

Gen-length:  699 , Token Count:  180


(cs336-basics) ➜  cs336_basics git:(main) ✗ python generate.py
prompt:  Once upon a time
Answer:  , there was a little girl named Sally. Sally loved to play outside in the sun. She liked to look at the big, orange sky. It was a clear, sunny day and the sun was shining bright.
One day, Sally saw a little bird stuck in a tree. She wanted to help the little bird. So, she had an idea. She took a long stick and began to bury the bird. She dug a hole in the ground and put the bird inside.
But then, something unexpected happened. The bird turned into a big, strong bird! The bird said, "Thank you for my help, Sally. I was stuck as a bird and could fly again. You helped me find a new home." Sally was happy and thanked the bird. She learned that even if she could not find a way to make a new friend.

Gen-length:  703 , Token Count:  173


(cs336-basics) ➜  cs336_basics git:(main) ✗ python generate.py
prompt:  Once upon a time
Answer:  , in a small town, there was a little girl named Lily. Lily loved to pick flowers in her garden every day. One day, she found a big, pretty flower. The flower was very pretty and it was always very pretty.
One day, Lily wanted to pick a flower to make a flower bloom. She thought about how to make the flower bloom. So, she picked a flower and put it in the big, bright flower. The flower bloomed into a flower.
The flower was very pretty. It bloomed and grew. It was so pretty that it made pretty flowers bloom. Lily and the flower were very happy. They knew that when the flower bloomed, it made them a beautiful flower. And from that day on, Lily and the flower were the best of friends.

Gen-length:  691 , Token Count:  160

```