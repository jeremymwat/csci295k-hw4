import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

word_to_int = dict()
int_to_word = dict()
training_corpus = []
test_corpus = []
word_set = set()
word_counter = 0

embedsz = 30
vocabsz = 7280
batchsz = 20
num_epoch = 1

def get_batches(location, input_list, batch_size):
    return input_list[location:location+batch_size], input_list[location+1:location+batch_size+1]

with open("train.txt", "r") as train_file:
    for line in train_file:
        line = line.rstrip()
        words = line.split(" ")
        for word in words:
            word = word.lower()
            if word in word_set:
                training_corpus.append(word_to_int[word])
            else:
                word_to_int[word] = word_counter
                int_to_word[word_counter] = word
                training_corpus.append(word_counter)
                word_counter += 1
                word_set.add(word)

with open("test.txt", "r") as train_file:
    for line in train_file:
        line = line.rstrip()
        words = line.split(" ")
        for word in words:
            word = word.lower()
            if word in word_set:
                test_corpus.append(word_to_int[word])
            else:
                word_to_int[word] = word_counter
                int_to_word[word_counter] = word
                test_corpus.append(word_counter)
                word_counter += 1
                word_set.add(word)


inpt = tf.placeholder(tf.int32, [None])
outpt = tf.placeholder(tf.int32, [None])

E = tf.Variable(tf.truncated_normal([vocabsz, embedsz], stddev=0.1))

embd = tf.nn.embedding_lookup(E,inpt)

hsize = 31

w1 = tf.Variable(tf.truncated_normal([embedsz, hsize], stddev=0.1))
b1 = tf.Variable(tf.zeros([hsize]))

w2 = tf.Variable(tf.truncated_normal([hsize,vocabsz],  stddev=0.1))
b2 = tf.Variable(tf.zeros([vocabsz]))

h1 = tf.nn.relu(tf.matmul(embd, w1)+b1)

logits = tf.matmul(h1, w2)+b2
error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, outpt))

train_step = tf.train.AdamOptimizer(0.0001).minimize(error)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for e in range(num_epoch):
    total_error = 0
    x = 0
    while x < len(training_corpus)-batchsz:
        inpt_tensor, outpt_tensor = get_batches(x, training_corpus, batchsz)
        x += batchsz
        _, err = sess.run([train_step, error], feed_dict={inpt:inpt_tensor, outpt:outpt_tensor})
        total_error += err*batchsz # multiply by batchsize to keep calculations for avg perplexity correct
        if x % 1000 == 0:
            print err
            print total_error
            print x
            print np.exp(total_error/(x))

batchsz = len(test_corpus)-1
input_tensor, output_tensor = get_batches(0, test_corpus, batchsz)
v = sess.run(error, feed_dict={inpt:input_tensor, outpt:output_tensor})

print np.exp(v)