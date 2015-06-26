import theano
import theano.tensor as T
import numpy as np

class memnn(object):
	"""A basic implementation of memory network"""

	def __init__(self, input_units, hidden_units,gamma,alpha):
		W = input_units * 3
		D = hidden_units
		V = input_units
		#self.V = vocab_size

		self.U_O = theano.shared((np.random.uniform(-1.0, 1.0,(D, W)) * 0.2).astype(np.float32))
		self.U_R = theano.shared((np.random.uniform(-1.0, 1.0,(D, W)) * 0.2).astype(np.float32))

		f1 = T.ivector("f1")
		_f1 = T.imatrix("_f1")

		f2 = T.ivector("f2")
		_f2 = T.imatrix("_f2")

		r = T.ivector("r")
		_r = T.imatrix("_r")

		x = T.ivector("x")
		m = T.imatrix("m") #memory
		v = T.imatrix("v") #vocabulary

		def S_O(x,y):
			x_emb = T.dot(self.U_O[:,:V],x)
			y_emb = T.dot(self.U_O[:,2*V:],y)
			return T.dot(x_emb.T,y_emb)

		def S_O_f(x,y):
			x_emb = T.dot(self.U_O[:,V:2*V],x)
			y_emb = T.dot(self.U_O[:,2*V:],y)
			return T.dot(x_emb.T,y_emb)

		def S_R(x,y):
			x_emb = T.dot(self.U_R[:,:V],x)
			y_emb = T.dot(self.U_R[:,2*V:],y)
			return T.dot(x_emb.T,y_emb)

		def S_R_f(x,y):
			x_emb = T.dot(self.U_R[:,V:2*V],x)
			y_emb = T.dot(self.U_R[:,2*V:],y)
			return T.dot(x_emb.T,y_emb)

		cost1,_ = theano.scan(
					lambda f_bar: T.largest(gamma - S_O(x,f1) + S_O(x,f_bar), 0),
					sequences = [_f1]
				)

		cost2,_ = theano.scan(
					lambda f_bar: T.largest(gamma - S_O(x,f2)  - S_O_f(f1,f2) + 
						S_O(x,f_bar) + S_O_f(f1,f_bar), 0),
					sequences = [_f2]
				)

		cost3,_ = theano.scan(
					lambda r_bar: T.largest(gamma - S_R(x,r) - S_R_f(f1,r)  - S_R_f(f2,r) +
						S_R(x,r_bar) +  S_R_f(f1,r_bar) + S_R_f(f2,r_bar), 0),
					sequences = [_r]
				)

		fact1 = T.argmax(S_O(x,m))
		self.getFact1 = theano.function(
					inputs= [x, m],
					outputs= fact1
					)

		fact2 = T.argmax(S_O(x,m) + S_O_f(f1,m))
		self.getFact2 = theano.function(
					inputs= [x, f1, m],
					outputs= fact2
					)

		predict = T.argmax(S_R(x,v) + S_R_f(f1,v) + S_R_f(f2,v))
		self.getAnswer = theano.function(
					inputs= [x, f1, f2, v],
					outputs= predict
					)

		cost = cost1.sum() + cost2.sum() + cost3.sum()

		grad_o, grad_r = T.grad(cost, [self.U_O,self.U_R])

		self.train = theano.function(
					inputs=[x, f1, _f1, f2, _f2, r, _r],
					outputs=[cost],
					updates=[(self.U_O, self.U_O - alpha*grad_o), (self.U_R,self.U_R - alpha*grad_r)]
					)

		self.computeCost = theano.function(
					inputs=[x, f1, _f1, f2, _f2, r, _r],
					outputs=[cost]
					)

	def train(self,x, f1, _f1, f2, _f2, r, _r):
		return self.train(x, f1, _f1, f2, _f2, r, _r)


	def predict(self, x_v, memory_indice, memory_v_1, R):
		# First Fact
		f1_predict = int(self.getFact1(x_v,memory_v_1.T))
		f1_v = memory_v_1[f1_predict]

		memory_v_2 = np.delete(memory_v_1, (f1_predict), axis = 0)

		# Second Fact
		f2_predict = int(self.getFact2(x_v,f1_v, memory_v_2.T))
		f2_v = memory_v_2[f2_predict]

		# Answer
		return int(self.getAnswer(x_v,f1_v,f2_v,R))

		