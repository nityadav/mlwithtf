import tensorflow as tf

def test_embedding_lookup():
	num = 4
	with tf.Session() as sess:
		A = tf.random_uniform([2], maxval=num, dtype=tf.int32)
		B = tf.truncated_normal([num + 1, 2, 2])
		C = tf.nn.embedding_lookup(B,A)
		[a,b,c] = (sess.run([A, B, C]))
		print "This is A"
		print(a)
		print "This is B"
		print(b)
		print "This is C"
		print(c)
		
def test():
	x = tf.Variable(0, name='x')
	model = tf.initialize_all_variables()
	with tf.Session() as session:
		session.run(model)
		for i in range(5):
			x = x + 1
			print(session.run(x))

test()