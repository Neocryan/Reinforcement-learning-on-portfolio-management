self.actor_grads = tf.gradients(self.actor_model.output, 
	actor_model_weights, -self.actor_critic_grad)

self.actor_model = self.actor_state_input, self.actor_model = self.create_actor_model()