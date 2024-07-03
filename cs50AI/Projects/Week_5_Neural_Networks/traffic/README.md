In this Project, I experimented with CNN using opencv. 
These are the things i tried:
    - Adjusting filer sizes: 
        - 2 by 2: 333/333 - 1s - 3ms/step - accuracy: 0.9719 - loss: 0.1568
        - 4 by 4: 333/333 - 1s - 4ms/step - accuracy: 0.9560 - loss: 0.2767
        - 3 by 3: 333/333 - 1s - 3ms/step - accuracy: 0.9320 - loss: 0.3196
> Found that the images have more localized features so smaller kernel size worked better
    - Increasing convolution layers:
        - used 3 convolution layer for results above
        - when using 4 convolution layers: 333/333 - 2s - 5ms/step - accuracy: 0.9581 - loss: 0.1876 for 2*2 kernel size. Increasing convolution layer didn't work
> Found that adding convolution layers doesn't always improve accuracy. But this did increase training time by a lot.
