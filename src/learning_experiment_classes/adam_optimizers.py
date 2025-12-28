"""
Adam and AdamW Optimizers for Min-Max problems (JAX-compatible).

These optimizers perform gradient descent on x_params and gradient ascent on y_params.
"""
import jax.numpy as jnp


class AdamWMinMax:
    """
    AdamW optimizer for Min-Max problems.
    
    Performs gradient descent on x_params and gradient ascent on y_params.
    Stores optimizer state (first and second moments) internally.
    
    Note: This uses Python lists for state management while keeping
    parameter arrays as JAX arrays. State updates are in-place via lists.
    """
    
    def __init__(self, x_params, y_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        """
        Args:
            x_params: Initial parameters for minimization (list of JAX arrays)
            y_params: Initial parameters for maximization (list of JAX arrays)
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay coefficient (L2 penalty)
        """
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        
        # Initialize state (First and Second moments) as JAX arrays
        self.state_x = [{'m': jnp.zeros_like(p), 'v': jnp.zeros_like(p)} for p in x_params]
        self.state_y = [{'m': jnp.zeros_like(p), 'v': jnp.zeros_like(p)} for p in y_params]

    def _apply_adamw(self, params, grads, states, maximize=False):
        """
        Internal AdamW step.
        
        Returns new_params list and updates states in-place.
        """
        beta1, beta2 = self.betas
        new_params = []
        
        # Bias correction factors based on current time step t
        bias_correction1 = 1.0 - beta1 ** self.t
        bias_correction2 = 1.0 - beta2 ** self.t

        for i, (p, g, s) in enumerate(zip(params, grads, states)):
            # 1. Update biased first moment estimate
            m_new = beta1 * s['m'] + (1 - beta1) * g
            
            # 2. Update biased second raw moment estimate
            v_new = beta2 * s['v'] + (1 - beta2) * (g ** 2)
            
            # Update state (in-place via dict mutation)
            states[i]['m'] = m_new
            states[i]['v'] = v_new
            
            # 3. Compute bias-corrected moments
            m_hat = m_new / bias_correction1
            v_hat = v_new / bias_correction2
            
            # 4. Compute the adaptive step
            step = m_hat / (jnp.sqrt(v_hat) + self.eps)
            
            # 5. Apply Weight Decay (Decoupled)
            p_decayed = p - (self.lr * self.wd * p)
            
            # 6. Apply Update
            if maximize:
                # Gradient Ascent: Add the step
                p_new = p_decayed + (self.lr * step)
            else:
                # Gradient Descent: Subtract the step
                p_new = p_decayed - (self.lr * step)
                
            new_params.append(p_new)
            
        return new_params

    def step(self, x_params, y_params, grads_x, grads_y, proj_y_fn=None):
        """
        Performs one step of optimization.
        
        Args:
            x_params: Current x parameters (list of JAX arrays)
            y_params: Current y parameters (list of JAX arrays)
            grads_x: Gradients for x (list of JAX arrays) - descent
            grads_y: Gradients for y (list of JAX arrays) - ascent
            proj_y_fn: Optional projection function for y params
            
        Returns:
            x_new: Updated x parameters
            y_new: Updated y parameters (projected if proj_y_fn provided)
        """
        self.t += 1
        
        # Update X (Minimization / Descent)
        x_new = self._apply_adamw(x_params, grads_x, self.state_x, maximize=False)
        
        # Update Y (Maximization / Ascent)
        y_unconstrained = self._apply_adamw(y_params, grads_y, self.state_y, maximize=True)
        
        # Project Y if projection function provided
        if proj_y_fn is not None:
            y_new = proj_y_fn(y_unconstrained)
        else:
            y_new = y_unconstrained
        
        return x_new, y_new


class AdamMinMax:
    """
    Adam optimizer for Min-Max problems (no weight decay).
    
    Performs gradient descent on x_params and gradient ascent on y_params.
    Stores optimizer state (first and second moments) internally.
    
    This is the standard Adam optimizer without the decoupled weight decay of AdamW.
    """
    
    def __init__(self, x_params, y_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """
        Args:
            x_params: Initial parameters for minimization (list of JAX arrays)
            y_params: Initial parameters for maximization (list of JAX arrays)
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to denominator for numerical stability
        """
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        
        # Initialize state (First and Second moments) as JAX arrays
        self.state_x = [{'m': jnp.zeros_like(p), 'v': jnp.zeros_like(p)} for p in x_params]
        self.state_y = [{'m': jnp.zeros_like(p), 'v': jnp.zeros_like(p)} for p in y_params]

    def _apply_adam(self, params, grads, states, maximize=False):
        """
        Internal Adam step (no weight decay).
        
        Returns new_params list and updates states in-place.
        """
        beta1, beta2 = self.betas
        new_params = []
        
        # Bias correction factors based on current time step t
        bias_correction1 = 1.0 - beta1 ** self.t
        bias_correction2 = 1.0 - beta2 ** self.t

        for i, (p, g, s) in enumerate(zip(params, grads, states)):
            # 1. Update biased first moment estimate
            m_new = beta1 * s['m'] + (1 - beta1) * g
            
            # 2. Update biased second raw moment estimate
            v_new = beta2 * s['v'] + (1 - beta2) * (g ** 2)
            
            # Update state (in-place via dict mutation)
            states[i]['m'] = m_new
            states[i]['v'] = v_new
            
            # 3. Compute bias-corrected moments
            m_hat = m_new / bias_correction1
            v_hat = v_new / bias_correction2
            
            # 4. Compute the adaptive step
            step = m_hat / (jnp.sqrt(v_hat) + self.eps)
            
            # 5. Apply Update (no weight decay)
            if maximize:
                # Gradient Ascent: Add the step
                p_new = p + (self.lr * step)
            else:
                # Gradient Descent: Subtract the step
                p_new = p - (self.lr * step)
                
            new_params.append(p_new)
            
        return new_params

    def step(self, x_params, y_params, grads_x, grads_y, proj_y_fn=None):
        """
        Performs one step of optimization.
        
        Args:
            x_params: Current x parameters (list of JAX arrays)
            y_params: Current y parameters (list of JAX arrays)
            grads_x: Gradients for x (list of JAX arrays) - descent
            grads_y: Gradients for y (list of JAX arrays) - ascent
            proj_y_fn: Optional projection function for y params
            
        Returns:
            x_new: Updated x parameters
            y_new: Updated y parameters (projected if proj_y_fn provided)
        """
        self.t += 1
        
        # Update X (Minimization / Descent)
        x_new = self._apply_adam(x_params, grads_x, self.state_x, maximize=False)
        
        # Update Y (Maximization / Ascent)
        y_unconstrained = self._apply_adam(y_params, grads_y, self.state_y, maximize=True)
        
        # Project Y if projection function provided
        if proj_y_fn is not None:
            y_new = proj_y_fn(y_unconstrained)
        else:
            y_new = y_unconstrained
        
        return x_new, y_new
