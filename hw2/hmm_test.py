import hmmlearn.hmm
import numpy as np

dataset = {

}

start_prob = np.array([0.8, 0.2], dtype=np.float)
transmat = np.array([
    [0.6, 0.4],
    [0.5, 0.5]
], dtype=np.float)
emiss_mat = np.array([
    [0.2, 0.4, 0.4],
    [0.5, 0.4, 0.1]
], dtype=np.float)

model = hmmlearn.hmm.MultinomialHMM(
    n_components=2,random_state=0,n_iter=0000,
    # params='ste',
    startprob_prior= start_prob,
    transmat_prior= transmat,
)

model.startprob_ = start_prob
model.transmat_ = transmat

model.emission_prob_= emiss_mat

test_samples = np.array([
    [2, 0, 2],
])

model.fit(test_samples)

# model.startprob_ = start_prob
# model.transmat_prior = transmat
# model.emissionprob_ = emiss_mat


print(model.score(test_samples))