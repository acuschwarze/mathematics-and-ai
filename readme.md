# Math 76.01: Mathematics and AI


<a name="course"></a>
<h2>
Course Description
</h2>
<p>Mathematics and AI offers an exploration of the intersection between mathematics and
artificial intelligence (AI). Covering state-of-the-art machine learning techniques and
their mathematical foundations, this course aims to provide students with both a broad theoretical
understanding and practical skills. The syllabus starts with a brief review of the history of AI,
and current limits and issues. This is followed by an introduction to statistical learning in a supervised setting and
a deeper dive on neural networks and their applications with some references to current
mathematical research. The syllabus continues with an overview of unsupervised learning methods and
their applications in feature selection. It concludes with student's presentations of their final projects.
</p>

<p>
<b>Prerequisite courses and skills</b>: Math  13, Math 22 or Math 24, and Math 23, or advanced placement/ instructor override.
Familiarity with at least one programming language. (Python preferred.)
Students who request an instructor override should have encountered the concepts in the <a href="#concepts">Prerequesite concepts</a> checklist in their previous coursework or self study.


</p>

<p><a name="concepts"></a><b>Prerequisite concepts:</b>
derivative of a function, chain rule, smooth function, optimization,
Taylor expansion, differential equation, fixed point,
vector, matrix, lines, curves, subspaces, eigenvector, eigenvalue
multivariate function, partial derivative, spherical coordinates,
probability distribution, conditional probability, joint probability
</p>

<p>
<b>Instructor</b>: Alice Schwarze (alice.c.schwarze@Dartmouth.edu)
</p>

<p>
<b>Classes</b>: (2) MWF 2:10 - 3:15 and x-hour Th 1:20 - 2:10
</p>

<a name="materials"></a>
<h2>
Textbooks and other materials
</h2>
<p>
<ul>
<li>
<i>Introduction to Statistical Learning with Applications in Python</i> by Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor<br>
available on the <a href="https://www.statlearning.com/">book's website</a>
</li>
<li>
<i>Data-Driven Science and Engineering</i> by Steven Brunton and Nathan Kutz<br>
videos and PDF (<a href="http://databookuw.com/databook.pdf">download</a>) available on the <a href="https://www.databookuw.com/">book's website</a>
</li>
<li>
		    <i>Geometry of Deep Learning</i> by Jong Chul Ye<br>
available via Dartmouth Libraries (<a href="https://search.library.dartmouth.edu/permalink/01DCL_INST/16rgcn8/alma991034027246105706">link</a>)
</li>
<li>
<i>Artificial Intelligence With an Introduction to Machine Learning</i> by Richard Neapolitan and Xia Jiang<br>
available via Dartmouth Libraries (<a href="https://search.library.dartmouth.edu/permalink/01DCL_INST/16rgcn8/alma991033334873005706">link</a>)
</li>
<li><i>Networks (2nd ed.)</i> by Mark Newman</li>
<li>
<i>Limits of AI - Theoretical, Practical, Ethical</i> by Klaus Mainzer and Reinhard Kahle<br>
		    available via Dartmouth Libraries (<a href="https://search.library.dartmouth.edu/permalink/01DCL_INST/134hn0f/cdi_proquest_ebookcentral_EBC31266916">link</a>)
</li>
</ul>
</p>

<a name="syllabus"></a>
<h2>Syllabus</h2>
<p>The following is a <strong>tentative</strong> schedule for the course. Please check back regularly for updates as the term progresses.</p>

<h4>Thu Jun 20</h4>
<h3>No class</h3>
<p class="smallp"></p>
<p class="smallp"></p>

<br><h4>Fri Jun 21</h4>
<h3>Lecture: Artificial intelligence: Ideas and their evolution</h3>
<p class="smallp"><b>Keywords:</b> Turing test, Dartmouth workshop, expert systems, strong AI, weak AI, artificial general intelligence (AGI)</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Neapolitan et al. Chapter 1.1 "History of Artificial Intelligence"
(<a href="https://ebookcentral.proquest.com/lib/dartmouth-ebooks/reader.action?docID=5321358&ppg=19">book chapter</a>)</li>
<li>Veisdal: "The Birthplace of AI" (<a href="https://www.privatdozent.co/p/the-birth-of-ai-1956">substack post</a>)</li>
<li>Turing &sect;1 "The Imitation Game" (<a href="https://www.csee.umbc.edu/courses/471/papers/turing.pdf">full paper</a>)</li>
</ul>
</p>

<br><h4>Mon Jun 24</h4>
<h3>Lecture: Representing knowledge</h3>
<p class="smallp"><b>Keywords:</b> tables, functions, frames, knowledge graphs, causal networks, directed acyclic graphs, Bayesian networks, Markov random fields (MRFs)</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Neapolitan et al. Chapter 4 "Certain Knowledge Representation"
(<a href="https://ebookcentral.proquest.com/lib/dartmouth-ebooks/reader.action?docID=5321358&ppg=92">book chapter</a>)</li>
<li>Neapolitan et al. Chapter 7 "Uncertain Knowledge Representation"
(<a href="https://ebookcentral.proquest.com/lib/dartmouth-ebooks/reader.action?docID=5321358&ppg=156">book chapter</a>)</li>
<li>Additional reading to review probability: Neapolitan et al. Chapter 6.1 "Probability Basics" and Chapter 6.2 "Random Variables"
(<a href="https://ebookcentral.proquest.com/lib/dartmouth-ebooks/reader.action?docID=5321358&ppg=126">book chapter</a>)</li>
</ul>
</p>

<br><h4>Wed Jun 26</h4>
<h3>Lecture: Linear regression</h3>
<p class="smallp"><b>Keywords:</b> linear regression, gradient descent, mean squared error</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>James et al. Chapter 3 "Linear Regression"
(<a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">link to full book download</a>)</li>
<li>Brunton et al. Section 1.4
(<a href="https://www.youtube.com/watch?v=PjeOmOz9jSY&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv">video 1</a>,
<a href="https://www.youtube.com/watch?v=02QCtHM1qb4&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv">video 2</a>,
<a href="https://www.youtube.com/watch?v=N9uf0YGDaWk&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv">video 3</a>,
<a href="https://www.youtube.com/watch?v=EDPCsD6BzWE&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv">video 4</a>
<a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Thu Jun 27</h4>
<h3>Lecture: Regression and classification</h3>
<p class="smallp"><b>Keywords:</b> logistic regression, k-nearest neighbors (KNN), linear discriminant analysis (LDA), quadratic discriminant analysis (QDA), naive Bayes, 1-hot encoding</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Brunton et al. Section 4.1 (<a href="https://youtu.be/32qEQ29QJco">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
<li>Brunton et al. Section 4.2 (<a href="https://youtu.be/YepA2w3EUEM">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
<li>James et al. Chapter 4 "Classification"
(<a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">link to full book download</a>)</li>
<li>Brunton et al. Section 5.6 (<a href="https://youtu.be/iUved-z75MY">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Fri Jun 28</h4>
<h3>Lecture: Resampling and validation</h3>
<p class="smallp"><b>Keywords:</b> Crossvalidation, bootstrap, data leakage</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>James et al. Chapter 5 "Resampling Methods"
(<a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">link to full book download</a>)</li>
<li>Brunton et al. Section 4.6 (<a href="https://youtu.be/NoQV1lc7OlU">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Mon Jul 1</h4>
<h3>Lecture: Feature selection</h3>
<p class="smallp"><b>Keywords:</b> subset selection, shrinkage, dimension reduction, principal component regression (PCR)</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>James et al. Chapter 6 "Linear Model Selection and Regularization"
(<a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">link to full book download</a>)</li>
<li>Brunton et al. Section 5.1 (<a href="https://youtu.be/2RTD5569m38">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Wed Jul 3</h4>
<h3>Lecture: Regularization</h3>
<p class="smallp"><b>Keywords:</b> ridge regression, lasso</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>James et al. Chapter 6 "Linear Model Selection and Regularization"
(<a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">link to full book download</a>)</li>
<li>Brunton et al. Section 3.5 (<a href="https://www.youtube.com/watch?v=GaXfqoLR_yI&list=PLMrJAkhIeNNRHP5UA-gIimsXLQyHXxRty">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
<li>Brunton et al. Section 4.3 (<a href="https://youtu.be/jnSi1Q3WVEw">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
<li>Brunton et al. Section 4.4 (<a href="https://youtu.be/c5ish_bQetE">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
<li>Brunton et al. Section 4.5 (<a href="https://youtu.be/sMEpJGKUYoE">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Thu Jul 4</h4>
<h3>No class on Independence Day.</h3>
<p class="smallp"></p>
<p class="smallp"></p>

<br><h4>Fri Jul 5</h4>
<h3>Lecture: Basis functions for regression</h3>
<p class="smallp"><b>Keywords:</b> step functions, splines, radial basis functions (RBFs), generalized additive models (GAMs)</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>James et al. Chapter 7 "Moving Beyond Linearity"
(<a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">link to full book download</a>)</li>
</ul>
</p>

<br><h4>Mon Jul 8</h4>
<h3>Lecture: Decision trees</h3>
<p class="smallp"><b>Keywords:</b> regression trees, classification trees, tree ensemble methods, bagging, boosting, random forests, Bayesian additive regression trees (BART)</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>James et al. Chapter 8 "Decision Trees"
(<a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">link to full book download</a>)</li>
<li>Brunton et al. Section 5.8 (<a href="https://youtu.be/fsE9gzbf8Z4">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Wed Jul 10</h4>
<h3>Lecture: Support vector machines</h3>
<p class="smallp"><b>Keywords:</b> maximum-margin models, hard margin, soft margin, VC theory, nonlinear kernels</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>James et al. Chapter 9 "Support Vector Machines"
(<a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">link to full book download</a>)</li>
<li>Brunton et al. Section 5.7 (<a href="https://youtu.be/NOKOJWQ2_iE">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Thu Jul 11</h4>
<h3>Lecture: Kernel methods</h3>
<p class="smallp"><b>Keywords:</b> kernel trick, kernel ridge regression, reproducing kernel Hilbert spaces, representer theorems</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>TBD</li>
</ul>
</p>

<br><h4>Fri Jul 12</h4>
<h3>Catch-up and Review</h3>
<p class="smallp"></p>

<br><h4>Mon Jul 15</h4>
<h3>Lecture: Introduction to neural networks: Perceptron and beyond</h3>
<p class="smallp"><b>Keywords:</b> perceptron, multi-class perceptron, universal approximation theorems, ReLU, softmax</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Brunton et al. Section 6.1 (<a href="https://youtu.be/v2EVAj0a8g8">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
<li>Brunton et al. Section 6.2 (<a href="https://youtu.be/Vj075U107MI">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Wed Jul 17</h4>
<h3>Lecture: Neural network architectures and neural coding</h3>
<p class="smallp"><b>Keywords:</b> feed-forward neural network, deep learning, encoder, decoder</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Brunton et al. Section 6.7 (<a href="https://youtu.be/D47ApuF5A7I">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Thu Jul 18</h4>
<h3>Lecture: Training and regularizing neural networks</h3>
<p class="smallp"><b>Keywords:</b> backpropagation, stochastic gradient descent, Adam, drop out</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Brunton et al. Section 6.3 (<a href="https://youtu.be/lq7f97HyyQg">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
<li>Brunton et al. Section 6.4 (<a href="https://youtu.be/_adhFSH66jc">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Fri Jul 19</h4>
<h3>Lecture: Transfer learning</h3>
<p class="smallp"><b>Keywords:</b> teacher-student learning, multitask learning</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>TBD</li>
</ul>
</p>

<br><h4>Mon Jul 22</h4>
<h3>Lecture: Forecasting and prediction</h3>
<p class="smallp"><b>Keywords:</b> Taken's theorem, time-delayed embedding, recurrent neural networks (RNNs), reservoir computing</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Brunton et al. Section 6.6 (<a href="https://youtu.be/JfeB_n4zsRM">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Wed Jul 24</h4>
<h3>Lecture: Natural language processing</h3>
<p class="smallp"><b>Keywords:</b> structured prediction, text classification, bag of words, self-supervised learning, word embeddings</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>TBD</li>
</ul>
</p>

<br><h4>Thu Jul 25</h4>
<h3>Lecture: Natural language processing (continued)</h3>
<p class="smallp"><b>Keywords:</b> long-term short-term memory, attention, transformer, generative pre-trained transformers (GPTs)</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>TBD</li>
</ul>
</p>

<br><h4>Fri Jul 26</h4>
<h3>Lecture: Image generation and more transfer learning</h3>
<p class="smallp"><b>Keywords:</b> general adversarial networks (GANs), contrastive language-image pre-training (CLIP), DALL-E, diffusion</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>TBD</li>
</ul>
</p>

<br><h4>Mon Jul 29</h4>
<h3>Project proposals</h3>
<p class="smallp"></p>
<p class="smallp"></p>

<br><h4>Wed Jul 31</h4>
<h3>Lecture: Representation learning</h3>
<p class="smallp"><b>Keywords:</b> latent space, autoencoders, restricted Boltzmann machines (RBMs)</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Brunton et al. Section 5.2 (<a href="https://youtu.be/Nt32lTUeCcM">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Thu Aug 1</h4>
<h3>No class</h3>
<p class="smallp"></p>
<p class="smallp"></p>

<br><h4>Fri Aug 2</h4>
<h3>Lecture: Principal component analysis</h3>
<p class="smallp"><b>Keywords:</b> principal component analysis (PCA), matrix factorizations, Hebbian learning</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Brunton et al. Section 1.1
(<a href="https://www.youtube.com/watch?v=nbBvuuNVfco&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
<li>Brunton et al. Section 1.5
(<a href="https://www.youtube.com/watch?v=fkf4IBRSeEc&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Mon Aug 5</h4>
<h3>Project updates</h3>
<p class="smallp"></p>
<p class="smallp"></p>

<br><h4>Wed Aug 7</h4>
<h3>Lecture: The topology of data</h3>
<p class="smallp"><b>Keywords:</b> self-organizing maps (SOMs), competitive learning, topological data analysis (TDA)</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>TBD</li>
</ul>
</p>

<br><h4>Thu Aug 8</h4>
<h3>No class</h3>
<p class="smallp"></p>
<p class="smallp"></p>

<br><h4>Fri Aug 9</h4>
<h3>Lecture: Clustering</h3>
<p class="smallp"><b>Keywords:</b> k-means clustering, hierarchical clustering</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Brunton et al. Section 5.3
(<a href="https://youtu.be/CMUe26KVhew">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
<li>Brunton et al. Section 5.4
(<a href="https://youtu.be/4B3dYs6t_qU">video</a>, <a href="http://databookuw.com/databook.pdf">book</a>)</li>
</ul>
</p>

<br><h4>Mon Aug 12</h4>
<h3>Project updates</h3>
<p class="smallp"></p>
<p class="smallp"></p>

<br><h4>Wed Aug 14</h4>
<h3>Lecture: Network analysis</h3>
<p class="smallp"><b>Keywords:</b> centrality measures, community detection, modularity maximization, belief propagation</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Newman Chapter 7.1</li>
<li>Newman Chapter 14</li>
</ul>
</p>

<br><h4>Thu Aug 15</h4>
<h3>No class</h3>
<p class="smallp"></p>
<p class="smallp"></p>

<br><h4>Fri Aug 16</h4>
<h3>Lecture: Matrix completion</h3>
<p class="smallp"><b>Keywords:</b> Low rank matrix completion, high rank matrix completion, link prediction, recommender systems</p>
<p class="smallp"><b>Reading material:</b>
<ul>
<li>Brunton et al. Section 1.5 (<a href="https://www.youtube.com/watch?v=sooj-_bXWgk&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv&index=9">video</a>)</li>
</ul>
</p>

<br><h4>Mon Aug 19</h4>
<h3>Final project presentations</h3>
<p class="smallp"></p>
<p class="smallp"></p>

<br><h4>Wed Aug 21</h4>
<h3>Final project presentations</h3>
<p class="smallp"></p>
<p class="smallp"></p>
<br>
