import random
import pickle
import os
import numpy as np
import pandas as pd



'''
This template serves as a starting point for your Part 1 agent.
'''


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        
        self.project_part = params['project_part'] 

        ### starting remaining inventory and inventory replenish rate are provided
        ## every time the inventory is replenished, it is set to the inventory limit
        ## the inventory_replenish rate is how often the inventory is replenished
        ## inventory is reset to a uniformly random integer from 7 to 20 every 20 customers.
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']

        self.filename = os.path.join(os.path.dirname(__file__), 'helper_files_here', 'purchase_model1.pkl') 
        #ALSO Note we need same model for purchase so we need to rerun ipyb folder if we ever change model, 
        # and we need to make sure the model is trained on the same price grid as the one used in this agent.
        value_path = os.path.join(os.path.dirname(__file__),'helper_files_here','value_table.pkl')


        with open(self.filename, 'rb') as file:
            self.trained_model = pickle.load(file)

        with open(value_path, 'rb') as file:
            self.V = pickle.load(file)
        
        self.price_grid = np.linspace(0, 500, 101)  
        # Baseline grid. #BEWARE TO CHNAGE THIS IN PART 2, AS IT WILL AFFECT THE VALUE TABLE. WE WILL MEED TO REBUILD THE VALUE TABLE IF WE CHANGE THE PRICE GRID.

    @staticmethod
    def get_single_2dstep_revenue_maximizing_price_and_revenue(Vtplus1_k_minus_1, Vtplus1_k, price_options, demand_predictions):
        p = np.array(price_options)
        d = np.array(demand_predictions)

        expected_revenue = d * (p + Vtplus1_k_minus_1) + (1 - d) * Vtplus1_k
        idx = np.argmax(expected_revenue)

        return float(p[idx]), float(expected_revenue[idx])


    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        '''
        This function updates your internal state based on the last sale that occurred.
        This template shows you several ways you can keep track of important metrics.
        '''
        ### keep track of who, if anyone, the customer bought from
        did_customer_buy_from_me = (last_sale[0] == self.this_agent_number)

        ### keep track of the prices that were offered in the last sale
        my_last_prices = last_sale[1][self.this_agent_number]

        ### keep track of the profit for this agent after the last sale
        my_current_profit = state[self.this_agent_number]

        ### keep track of the inventory levels after the last sale
        self.remaining_inventory = inventories[self.this_agent_number]

        ### keep track of the time until the next replenishment
        time_until_replenish = time_until_replenish

        ### TODO - add your code here to potentially update your pricing strategy 
        ### based on what happened in the last round

        # hmm by update pricing strategy, do you mean to update the model or just the price grid?


        pass

    def action(self, obs):
        '''
        This function is called every time the agent needs to choose an action by the environment.

        The input 'obs' is a 5 tuple, containing the following information:
        -- new_buyer_covariates: a vector of length 3, containing the covariates of the new buyer.
        -- last_sale: a tuple of length 2. The first element is the index of the agent that made the last sale, if it is NaN, then the customer did not make a purchase. The second element is a numpy array of length n_agents, containing the prices that were offered by each agent in the last sale.
        -- state: a vector of length n_agents, containing the current profit of each agent.
        -- inventories: a vector of length n_agents, containing the current inventory level of each agent.
        -- time_until_replenish: an integer indicating the time until the next replenishment, by which time your (and your opponent's, in part 2) remaining inventory will be reset to the inventory limit.

        The expected output is a single number, indicating the price that you would post for the new buyer.
        '''

        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        k = self.remaining_inventory
        t = self.inventory_replenish - time_until_replenish

        if k == 0:
            return float(self.price_grid[-1])

        features = pd.DataFrame(np.tile(new_buyer_covariates, (len(self.price_grid), 1)),
                                        columns=['Covariate1', 'Covariate2', 'Covariate3'])
        features['price_item'] = self.price_grid
        probability = np.clip(self.trained_model.predict(features), 0, 1)
        price, _ = self.get_single_2dstep_revenue_maximizing_price_and_revenue(self.V[t + 1, k - 1], self.V[t + 1, k],self.price_grid,probability)
        return price
