This directory provides a minimal experimental model to test the claim of an article of McCandlish, Kaplan, Amodei, Openai 'An Empirical Model of Large-Batch Training', https://arxiv.org/pdf/1812.06162 

The article gives recepe of an optimal choise of batch size and learning rate during training, after which resources spent give diminishing returm during training, deriving an optimal batch size for the training and optimal learning rate from theoretical model. The theoretical dependence of optimal learning rate on batch size is 

$$ 
\epsilon_{\text{opt}}(B) = \frac{\epsilon_{\text{max}}}{1+B_{\text{noise}}/B},
$$

where $\epsilon_{\text{max}}​= ..​.$ and $B_{\text{noise}} = ...$  

The implementation tests the theoretical model in several models, starting from MNIST, via the following protocol: 

1. Optimal learning rate (step size providing maximal loss drop) is found for fixed batch size via calculating loss drop at a fixed value of parameters for different update steps and then step size delivering the maximal loss drop is found

2. ...

Testing protocol:

1. MNIST: 
we define loss as ...

    a. Optimal learning rate: 

For a fixed value of parameters (weights) of a model $\theta_{t_n}$
    for each batch size B do 
        for each learning rate (update step) $\epsilon_i $
            Sample data samples of the batch size X_B = {x_1.. x_B}
            Calculate 
            - update direction vector (vector in which direction to move): 
            $G_{est} = E_{\text{batch}} [ \nabla_{\theta_{t_n}} L(x, \theta_{t_n})]   =  \frac{1}{B} \sum_{x \in \{x_1.. x_B\}} \nabla_{\theta} L(x, \theta) |_{\theta = \theta_{t_n}} $ 
            - loss change with update step: 
            $\Delta L (\theta_{t_n}, \epsilon_i, X_B) = L(\theta_{t_n} + \epsilon_i G_{est} ) - L(\theta_{t_n})$
        From a set of values $\{ \Delta L (\epsilon_i) \}$ find epsilon that minimizes the loss change $\epsilon_{\text{opt}}(B) = argmin\{ \Delta L (\epsilon_i) \}$
    Check that the resulting $\epsilon_{\text{opt}}(B)$ follows the equation for theoretical $\epsilon_{opt}(B)$

Specific values of parameters: 