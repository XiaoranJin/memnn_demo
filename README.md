# memnn_demo

A theano-based project to reproduce facebook's memory network.

So far, I only managed to implement the basic memory network with two supporting facts and no time feature in consider. Runing on the bAbI datasets (https://research.facebook.com/researchers/1543934539189348) only gave a F1 score around 0.6. 

I am hoping take the write time into account would give a siginicant improvement towards the 100% accuracy as shown in facebook’s paper. 

REFERENCES
> Weston J, Bordes A, Chopra S, et al. Towards AI-complete question answering: a set of prerequisite toy tasks[J]. arXiv preprint arXiv:1502.05698, 2015.

> Weston J, Chopra S, Bordes A. Memory networks[J]. arXiv preprint arXiv:1410.3916, 2014.