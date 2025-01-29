Hello, below will be my explanation to how I set up all the maths in this.


I will first start by stating how numpy defines the shape of a matrix.

In linear algebra, a n x m matrix is a matrix with n columns and m rows, ex. below:


                            n = 4
                       [[1, 2, 3, 4]
                m = 3   [4, 5, 6, 7]
                        [8, 9, 8, 7]]

In linear algebra, we are used to seeing this matrix be defined by first stating n, and then m in a n x m fasion, for the example above it would be a 4 x 3 matrix.
It is important to note that numpy will state the shape of the above matrix as (m, n), which in this case would be (3, 4), 
in the opposite direction as would normally be the case if written from a mathematical point of view.
This change however makes sense considering how these lists are initalised, lists are usually initalised horizontally, while vectors in maths are vertical,
this change is the cause for why numpy treats this shape the way it does.


My input shape:
Each image is stored in the below way:
[p1, p2, p3 ... p783, p784]

Where pi denotes 1 pixel in the 28 * 28 image, totalling a total of 784 pixels. 
I did not seperate these pixels based on rows and columns as there is no need, the network will figure out which pixel corresponds to what as it learns.
All of these are 1d arrays, of shape (784,) according to numpy.
Because the use of batches helps generalise the inputs and avoid complications due to outliers, we will store these 1d arrays inside of another array.
If we wanted to have 10 images in a batch, the array would look something as below:

[[p1, p2, p3 ... p783, p784]
 ... (x8)
 [p1, p2, p3 ... p783, p784]]

With the shape of (10, 784,) according to numpy.
To generalise this information, we can consider the input shape to be (batch size, number of inputs)

Weights and Biases:
The function of each layer looks something like the following:

f(x) = a(Wx + B)

where a is the activation function, W is the matrix of weights, and B is the vector of biases
We will look at the weights and biases specifically to determine what shape each one needs.

Running a quick test in numpy, you can quickly confirm that if you have 2 matrixes:

A, with shape (a, b)
and
B, with shape (c, d)

The matrix multiplication of AB has a shape (a, d)

AKA  (a, b) @ (c, d) => (a, d)

Based on the function, A would be our weights and B would be our batch in this case.
The weights should not affect batch size, so our "d" value would have to imply it. 
This can be confirmed in a quick simulation if you would like to confirm this.
Because of this, "a" would have to imply the size of the output vector.
Based on how matrix multiplication works, "b" = "c", where these 2 variables imply the amount of inputs.

Using this information we can define the function as:

f(x) = (output amount, input amount) @ (input amount, batch size)


However because of this, there is 1 small issue.
If you check above again, you will see that we are currently taking in the input in the shape (batch size, input amount), the transpose of what we want.
There is another issue, and that is that the current output will look like (output amount, batch size), which is fine at a glance, 
, but if you think about it, it directly means that you will have output amount arrays, each of them storing values from every batch.
This is the opposite (transpose) of what we want, we would prefer to have batch amount of arrays, each one of them storing all the outputs for that batch.

The solution to this is to make use of transposes, and how they interact with each other.
Since we get the inputs in transpose form, and want the output in transpose form, it would be be best to make the weights also be in transpose form.
We can develop this function using the below rule:

(AB).T = (B.T) * (A.T)
To reword this function,
output.T = Input @ Weights.T

Because of this, the final function is as follows:

f(x) = (batch size, input amount) @ (input amount, output amount)
f(x) = Input @ Weights

Note that because of the rule above, you want the inputs to be on the left now rather then the weights.

The bias has to have a shape which can be added to the shape (batch size, output amount).
Because of this, the shape of the bias will simply be (output amount).
Another result of all this is that even though we are making use of how transposes work, we don't actually have to ever calculate it, it will rather just work due to our design.