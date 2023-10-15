$$
\begin{split}
Loss & = k1.\text{Reconstruction Loss} + k2.\text{KL Divergence} \\
 & = 
\end{split}
$$

<!-- 
$$
\newcommand{\vect}[1]{\boldsymbol{#1}}

\mathcal{F}(\theta,\phi,k_1,k_2,\vect{x},\vect{z})=k_1.\mathop{\mathbb{E}}_{\vect{z}\sim q_{\phi(\vect{z}|\vect{x})}}[p_\theta(\vect{x}|\vect{z})]-k_2.D_{kl}(q_\theta(\vect{z}|\vect{x})||p(\vect{z}))
$$ -->

$$
\newcommand{\vect}[1]{\boldsymbol{#1}}

\mathcal{F}(\theta,\phi,k_1,k_2,\vect{x},\vect{z})=k_1.\mathop{\mathbb{E}}_{\vect{z}\sim q_{\phi(\vect{z}|\vect{x})}}[p_\theta(\vect{x}|\vect{z})]-k_2.KL(q_\theta(\vect{z}|\vect{x})||p(\vect{z}))
$$