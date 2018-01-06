import tensorflow as tf
import numpy as np
import sys

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    return tf.Variable(tf.zeros(shape), name=name)

class RecommenderModel(object):
    def __init__(self, num_users, num_items, num_features=20, reg_lambda=1e-5):
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)
        self.build_graph()
    
    def build_graph(self):
        self.u_idx = tf.placeholder(tf.int32, [None])
        self.v_idx = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.float32, [None])

        self.U = weight_variable([self.num_users, self.num_features], 'U')
        self.V = weight_variable([self.num_items, self.num_features], 'V')
        self.U_bias = bias_variable([self.num_users], 'U_bias')
        self.V_bias = bias_variable([self.num_items], 'V_bias')

        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)
        self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        self.r_hat = tf.add(self.r_hat, self.V_bias_embed)
        # self.r_hat = tf.add(self.r_hat, tf.reduce_mean(self.r))
        self.r_predict = tf.add(
            tf.matmul(self.U, tf.transpose(self.V)),
            tf.matmul(tf.reshape(self.U_bias, [self.num_users, 1]), tf.transpose(tf.reshape(self.V_bias, [self.num_items, 1]))))

        self.l2_loss = tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = tf.add(self.l2_loss, self.reg)

        self.optimizer = tf.train.AdamOptimizer()
        self.train_step = self.optimizer.minimize(self.reg_loss)

    def train_batch(self, session, train_u_idx, train_v_idx, train_r):
        feed_dict = {self.u_idx:train_u_idx, self.v_idx:train_v_idx, self.r:train_r}
        session.run(self.train_step, feed_dict)

        return session.run(self.l2_loss, feed_dict)
    
    def recommend(self, session, user_idx, result_count):
        _, item_list = session.run(tf.nn.top_k(self.r_predict[user_idx], result_count))
        return item_list

    def predict(self, session, u_idx, v_idx):
        feed_dict = {self.u_idx:[u_idx], self.v_idx:[v_idx]}

        return session.run(self.r_hat, feed_dict)

    def get_similar_products(self, session, item_index, result_count):
        #Get the normalized product parameters
        V_normalized = session.run(tf.nn.l2_normalize(self.V, 1))

        #Get the parameters for this product
        this_params_normalized = V_normalized[item_index]

        cos_similarity = tf.reduce_sum(tf.multiply(V_normalized,this_params_normalized), 1)
        top_similar = tf.nn.top_k(cos_similarity, result_count)
        score_list, idx_list = session.run(top_similar)

        return idx_list
    
    def save(self, session):
        saver = tf.train.Saver()
        saver.save(session, "./model.ckpt")

    def restore(self, session):
        saver = tf.train.Saver()
        saver.restore(session, "./model.ckpt")

class MovieLensDataLoader(object):
    def __init__(self):
        self.num_users = 0
        self.num_items = 0
        self.num_ratings = 0

        with open('./ml-100k/u.info', 'r') as f:
            for line in f.readlines():
                tokens = line.split()
                value = int(tokens[0])
                label = tokens[1]

                if label == "users":
                    self.num_users = value
                elif label == "items":
                    self.num_items = value
                elif label == "ratings":
                    self.num_ratings = value
        
        self.data_file = open("./ml-100k/u.data", "r")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.data_file.close()

    def load_next_batch(self, batch_size):
        user_idx = np.zeros([batch_size])
        item_idx = np.zeros([batch_size])
        rating = np.zeros([batch_size])
        
        for index in range(0, batch_size):
            line = self.data_file.readline()
            if line == "":
                self.data_file.seek(0)
                line = self.data_file.readline()
            
            parts = line.split()
            user_idx[index] = float(parts[0]) - 1
            item_idx[index] = float(parts[1]) - 1
            rating[index] = float(parts[2])
            
        return (user_idx, item_idx, rating)


def train():
    with MovieLensDataLoader() as loader:
        print "Users:", loader.num_users
        print "Items:", loader.num_items
        print "Ratings:", loader.num_ratings

        with tf.Session() as session:
            model = RecommenderModel(loader.num_users, loader.num_items, num_features=2000)

            init_op = tf.global_variables_initializer()
            session.run(init_op)

            for step in range(0, 3500):
                train_user_idx, train_item_idx, train_rating = loader.load_next_batch(200)
                loss = model.train_batch(session, train_user_idx, train_item_idx, train_rating)
                
                if step % 10 == 0:
                    print "Step:", step, "Loss:", loss
            model.save(session)

def recommend(user_idx):
    with MovieLensDataLoader() as loader:
        with tf.Session() as session:
            model = RecommenderModel(loader.num_users, loader.num_items, num_features=2000)

            model.restore(session)

            recommended_items = model.recommend(session, user_idx, 5)

            print "Recommended:", recommended_items

def predict_rating(user_idx, item_idx):
    with MovieLensDataLoader() as loader:
        with tf.Session() as session:
            model = RecommenderModel(loader.num_users, loader.num_items, num_features=2000)

            model.restore(session)

            rating = model.predict(session, user_idx, item_idx)

            print "Predicted rating:", rating

def similar_items(item_idx):
    with MovieLensDataLoader() as loader:
        with tf.Session() as session:
            model = RecommenderModel(loader.num_users, loader.num_items, num_features=2000)

            model.restore(session)

            index_list = model.get_similar_products(session, item_idx, 5)

            print index_list

if len(sys.argv) == 1:
    print "Usage: [--train] [--recommend user_index] [--similar item_index] [--predict user_index item_index]"
elif sys.argv[1] == "--train":
    train()
elif sys.argv[1] == "--recommend":
    recommend(int(sys.argv[2]))
elif sys.argv[1] == "--similar":
    similar_items(int(sys.argv[2]))    
elif sys.argv[1] == "--predict":
    predict_rating(int(sys.argv[2]), int(sys.argv[3]))    
