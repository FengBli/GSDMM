## Motivation
When I was working on the third homework of _data mining_ course: clustering the short texts, I found this paper in Reference section which turned out be to the one recommended by Mr. Zhang in class. So I tried to implement the GSDMM algorithm proposed myself, of course, with the help of online resources.

## NOTICE
This implementation is still on going.

## Data Format
- `vacabulary.json`, with one word and its corresponding id each line.
- `train_tokens.json`, with one doc-id and its token list each line.
- `train_topics.json`, using for validation.

## Reference
- Paper
 - Yin, J. and Wang, J., 2014, August. _A dirichlet multinomial mixture model-based approach for short text clustering._ In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 233-242).ACM.
 - Nguyen, D. Q., Billingsley, R., Du, L., & Johnson, M. (2015). _Improving topic models with latent feature word representations._ , 3, 299-313.
- Code
- [datquocnguyen/jLDADMM](https://github.com/datquocnguyen/jLDADMM): java version
 - [atefm/pDMM](https://github.com/atefm/pDMM): python version
