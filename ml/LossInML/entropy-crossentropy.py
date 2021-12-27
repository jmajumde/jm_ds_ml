
# 1. https://machinelearningmastery.com/what-is-information-entropy/
# 2. https://machinelearningmastery.com/cross-entropy-for-machine-learning/

'''
We know the probability of rolling any number is 1/6, which is a smaller number
than 1/2 for a coin flip, therefore we would expect more surprise or a larger amount of information.
'''
# calculate the information for a dice roll
from math import log2
# probability of the event
p = 1.0 / 6.0
# calculate information for event
h = -log2(p)
# print the result
print('p(x)=%.3f, information: %.3f bits' % (p, h))

'''
To compare prob v/s information we can create a plot to showcase lower prob events are more
surprising, compared to high probability events and carry more information. High probability
events are less surprising hence carry less information
'''
from math import log2
import matplotlib.pyplot as plt
# list of probabilities
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# calculate information
info = [-log2(p) for p in probs]
plt.plot(probs, info, marker=".")
plt.title("probability v/s information")
plt.xlabel("probability")
plt.ylabel("Information")
plt.show()



# calculate entropy for a random variable
'''
The intuition for entropy is that it is the average number of bits required to
represent or transmit an event drawn from the probability distribution for the random variable.
Entropy can be calculated for a random variable X with k in K discrete states as follows:

H(X) = -sum(each k in K p(k) * log(p(k)))
That is the negative of the sum of the probability of each event multiplied by the log
of the probability of each event.

Like information, the log() function uses base-2 and the units are bits.
A natural logarithm can be used instead and the units will be nats.

The lowest entropy is calculated for a random variable that has a single event with
a probability of 1.0, a certainty. The largest entropy for a random variable will be
if all events are equally likely.

We can consider a roll of a fair die and calculate the entropy for the variable.
Each outcome has the same probability of 1/6, therefore it is a uniform probability distribution.
We therefore would expect the average information to be the same information for a
single event calculated in the previous section.
'''
from math import log2
n = 6
# prob of one event
p = 1.0/n
entropy = -sum([p * log2(p) for _ in range(n)])
print('entropy: %.3f bits' % entropy)


'''
In the case where one event dominates, such as a skewed probability distribution,
then there is less surprise and the distribution will have a lower entropy.
In the case where no event dominates another, such as equal or approximately equal
probability distribution, then we would expect larger or maximum entropy.

- Skewed Probability Distribution (unsurprising): Low entropy.
- Balanced Probability Distribution (surprising): High entropy.

If we transition from skewed to equal probability of events in the distribution
we would expect entropy to start low and increase, specifically from the lowest
entropy of 0.0 for events with impossibility/certainty (probability of 0 and 1 respectively)
to the largest entropy of 1.0 for events with equal probability.

The example below implements this, creating each probability distribution in this transition,
calculating the entropy for each and plotting the result.
'''
# compare probability distributions vs entropy
from math import log2
from matplotlib import pyplot

# calculate entropy
def entropy(events, ets=1e-15):
	return -sum([p * log2(p + ets) for p in events])

# define probabilities
probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# create probability distribution
dists = [[p, 1.0 - p] for p in probs]
# calculate entropy for each distribution
ents = [entropy(d) for d in dists]
# plot probability distribution vs entropy
pyplot.plot(probs, ents, marker='.')
pyplot.title('Probability Distribution vs Entropy')
pyplot.xticks(probs, [str(d) for d in dists])
pyplot.xlabel('Probability Distribution')
pyplot.ylabel('Entropy (bits)')
pyplot.show()

'''
As expected, we can see that as the distribution of events changes from skewed to balanced,
the entropy increases from minimal to maximum values. That is, if the average event drawn from a
probability distribution is not surprising we get a lower entropy, whereas if it is surprising,
we get a larger entropy.

We can see that the transition is not linear, that it is super linear.
We can also see that this curve is symmetrical if we continued the transition to [0.6, 0.4] and
onward to [1.0, 0.0] for the two events, forming an inverted parabola-shape.
'''


##################3
'''
Cross-entropy builds upon the idea of entropy from information theory and calculates the number
of bits required to represent or transmit an average event from one distribution compared to
another distribution.

The intuition for this definition comes if we consider a target or underlying
probability distribution P and an approximation of the target distribution Q, then the
cross-entropy of Q from P is the number of additional bits to represent an event using Q instead of P.

The cross-entropy between two probability distributions, such as Q from P,
can be stated formally as:
    H(P, Q)
Where H() is the cross-entropy function, P may be the target distribution and
Q is the approximation of the target distribution.

Cross-entropy can be calculated using the probabilities of the events from P and Q, as follows:

H(P, Q) = – sum x in X P(x) * log(Q(x))
Where P(x) is the probability of the event x in P, Q(x) is the probability of event x in Q
and log is the base-2 logarithm, meaning that the results are in bits. If the base-e or natural logarithm is used instead, the result will have the units called nats.


'''
# implement





##################
# cross-entropy as loss function
##################
from math import log
from numpy import mean
def cross_entropy(p,q):
    return -sum([p[i]*log(q[i]) for i in range(len(p))])

#Consider a two-class classification task with the following 10 actual class labels (P) and
# predicted class labels (Q).
p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]

# calculate cross entropy for each example
results = list()
for i in range(len(p)):
	# create the distribution for each event {0, 1}
	expected = [1.0 - p[i], p[i]]
	predicted = [1.0 - q[i], q[i]]
	# calculate cross entropy for the two events
	ce = cross_entropy(expected, predicted)
	print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (p[i], q[i], ce))
	results.append(ce)
# calculate average crocross_entropy
mean_ce = mean(results)
print('\nAverage Cross Entropy: %.3f nats' % mean_ce)

# calculate cross-entropyusing keras
from numpy import asarray
from keras import backend
from keras.losses import binary_crossentropy

p = asarray([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
q = asarray([0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3])
# convert to keras variables
y_true = backend.variable(p)
y_pred = backend.variable(q)
mean_ce = backend.eval(binary_crossentropy(y_true,y_pred))
print(" Average Cross Entropy: %.3f nats" % mean_ce)















































###
