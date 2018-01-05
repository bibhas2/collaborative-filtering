# Collaborative Filtering Using Tensorflow

This is an implementation of NetFlix competition winning collaborative filtering as described in [here](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf).

The work was greatly inspired by [this other implementation](https://github.com/arongdari/MatrixFactorization-TensorFlow).

We use MovieLens data for training.

Run the code as:

```
python model.py
```

With 2000 features for each user and movie we get excellent accuracy after a few minutes of training.

## Dealing With Sparse Matrix
Very few users give any rating for most products. (Rating can be explicit like star rating or implicit like visiting a product's page).
As a result the training data is very sparse. When calculating cost (or, loss) in collaborative filtering we only take
into account ratings that are known for a product/user combination. This is not like a normal
neural network where all features play a role in cost calculation. Calculating cost can become tricky. The real hero here is
``tf.nn.embedding_lookup()``. This saves us a ton of trouble. 

Let's inspect the various matrices that are members of the ``RecommenderModel`` class.

``U`` is the user parameter (or, weight) matrix of ``num_users X num_features`` dimension. ``V`` is the item (product or movie) parameter
  matrix of ``num_items X num_features`` dimension. ``u_idx`` is a vector containing user indices in a training batch. This corresponds to the first column in the ``u.data`` file. It has a dimension of ``t X 1`` where ``t`` is the number of samples in a training batch. ``v_idx`` is a vector containing indices for items and corresponds to the second column of the ``u.data`` file.

 Then we do this:

```python
self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)
```

``U_embed`` is a subset of ``U`` corresponding to the users listed in ``u_idx``. It will be of dimension ``t X num_features``. ``V_embed`` is a subset of ``V`` corresponding to the items listed in ``v_idx``. It will also be of dimension ``t X num_features``.

The coolest thing happens now. We can get the predicted rating by doing element wise multiplication of ``U_embed`` and ``V_embed`` and summing up the rows.

```python
self.r_hat = tf.reduce_sum(
    tf.multiply(self.U_embed, self.V_embed), 
    reduction_indices=1)
```

Note: ``tf.multiply()`` does element wise multiplication. Where as ``tf.matmul()`` does a dot product.

``r_hat`` - the prdicted rating - will be ``t X 1`` vector.

We then add biases to ``r_hat`` as per the paper.

Calculating cost becomes super easy. We square the difference between the rating in training data ``r`` and the prediction ``r_hat`` and sum it all up to get the loss.

```python
self.l2_loss = tf.nn.l2_loss(
    tf.subtract(self.r, self.r_hat))
```

We then add regularization to the cost.