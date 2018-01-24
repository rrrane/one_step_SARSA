
#include <stdio.h>
#include <string.h>


#include "rlglue/Agent_common.h" /* Required for RL-Glue */
#include "utils.h"
#include "common.h"

//#define EPSILON		0.1 /* Exploration probability */
//#define	ALPHA		0.5 /* Learning rate */
//#define GAMMA		1.0 /* Discounting factor */
//#define Q_TERMINAL	0.0 /* Q-Value for terminal state */


static gsl_vector* local_action;
static action_t* this_action;
static gsl_vector* last_observation;

static double Q[GRIDWIDTH][GRIDHEIGHT][N_ACTIONS];

static const double epsilon = 0.1;
static const double alpha = 0.5;
static const double disc_factor = 1.0;
static const double q_terminal = 0.0;

void agent_init()
{

	//Allocate Memory

	local_action = gsl_vector_calloc(1);
	this_action = local_action;
	last_observation = gsl_vector_calloc(2);
	
	for(int i = 0; i < GRIDWIDTH; i++){
		for(int j = 0; j < GRIDHEIGHT; j++){
			for(int k = 0; k < N_ACTIONS; k++)
				Q[i][j][k] = 0;

		}
	}
	
}

const action_t *agent_start(const observation_t *this_observation) {
 	
  	//Read State

	int x = (int)gsl_vector_get(this_observation, 0);
	int y = (int)gsl_vector_get(this_observation, 1);
	

	//Get optimal and suboptimal actions
	int opt_actions[N_ACTIONS] = {0, 0, 0, 0, 0, 0, 0, 0};
	int sub_opt_actions[N_ACTIONS] = {0, 0, 0, 0, 0, 0, 0, 0};

	int maxcount = 0;
	int subcount = 0;
	
	double maxval = Q[x][y][0];
	for(int i = 1; i < N_ACTIONS; i++){
		if(maxval < Q[x][y][i])
			maxval = Q[x][y][i];
	}

	for(int i = 0; i < N_ACTIONS; i++){
		if(Q[x][y][i] == maxval)
			opt_actions[maxcount++] = i;
		else
			sub_opt_actions[subcount++] = i;
	}
	

	//Randomly select optimal action
	int maxindex = randInRange(maxcount);
	int opt_act = opt_actions[(maxindex < N_ACTIONS) ? maxindex : N_ACTIONS - 1];

	//Randomly select suboptimal action
	int subindex = randInRange(subcount);
	int sub_opt_act = sub_opt_actions[(subindex < N_ACTIONS) ? subindex : N_ACTIONS - 1];
	
	//Select action based on epsilon-greedy policy
	double p = rand_un();

	int act = randInRange(N_ACTIONS);

	act = (act < N_ACTIONS) ? act : N_ACTIONS - 1;
	
	if(p >= epsilon || subcount == 0)
		act = opt_act;
	else
		act = sub_opt_act;
	
	//Save action in local_action
	gsl_vector_set(local_action, 0, act);

	//Save last observation
	gsl_vector_memcpy(last_observation, this_observation);

  	return this_action;
}

const action_t *agent_step(double reward, const observation_t *this_observation) {


  	//Read State
	int x_prime = (int)gsl_vector_get(this_observation, 0);
	int y_prime = (int)gsl_vector_get(this_observation, 1);
	
	
	//Get set of optimal and suboptimal actions

	int opt_actions[N_ACTIONS] = {0, 0, 0, 0, 0, 0, 0, 0};
	int sub_opt_actions[N_ACTIONS] = {0, 0, 0, 0, 0, 0, 0, 0};


	int maxcount = 0;
	int subcount = 0;

	double maxval = Q[x_prime][y_prime][0];
	
	for(int i = 1; i < N_ACTIONS; i++){
		if(maxval < Q[x_prime][y_prime][i])
			maxval = Q[x_prime][y_prime][i];

	}

	for(int i = 0; i < N_ACTIONS; i++){
		if(Q[x_prime][y_prime][i] >= maxval)
			opt_actions[maxcount++] = i;
		else
			sub_opt_actions[subcount++] = i;
	}
	
	//Randomly select optimal action
	int maxindex = randInRange(maxcount);
	int opt_act = opt_actions[(maxindex < N_ACTIONS) ? maxindex : N_ACTIONS - 1];

	//Randomly select suboptimal action
	int subindex = randInRange(subcount);
	int sub_opt_act = sub_opt_actions[(subindex < N_ACTIONS) ? subindex : N_ACTIONS - 1];
	

	//Select action based on epsilon-greedy policy
	double p = rand_un();

	int act_prime = randInRange(N_ACTIONS);

	act_prime = (act_prime < N_ACTIONS) ? act_prime : N_ACTIONS - 1;
	
	if(p >= epsilon || subcount == 0)
		act_prime = opt_act;
	else
		act_prime = sub_opt_act;
	

	//Get last action
	int act = (int)gsl_vector_get(local_action, 0);
  	
	//Get last observation
	int x = (int)gsl_vector_get(last_observation, 0);
	int y = (int)gsl_vector_get(last_observation, 1);

	//Update action-value estimation Q
	
#ifdef SARSA
	Q[x][y][act] += alpha * (reward + disc_factor * Q[x_prime][y_prime][act_prime] - Q[x][y][act]);
#endif

#ifdef EXPECTED_SARSA
	double sum = 0.0;

	double pi[N_ACTIONS] = {0, 0, 0, 0, 0, 0, 0, 0};

	for(int i = 0; i < maxcount; i++)
		pi[opt_actions[i]] = (1 - epsilon)/(double)maxcount;

	for(int i = 0; i < subcount; i++)
		pi[sub_opt_actions[i]] = epsilon/(double)subcount;

	for(int i = 0; i < N_ACTIONS; i++)
		sum += pi[i]*Q[x_prime][y_prime][i];

	Q[x][y][act] += alpha * (reward + disc_factor * sum - Q[x][y][act]);
#endif
	//Save action in local_action
	gsl_vector_set(local_action, 0, act_prime);

	//Save last observation
	gsl_vector_memcpy(last_observation, this_observation);

  	return this_action;
}


void agent_end(double reward) {
  /* final learning update at end of episode */
	
	//Get last action
	int act = (int)gsl_vector_get(local_action, 0);
  	
	//Get last observation
	int x = (int)gsl_vector_get(last_observation, 0);
	int y = (int)gsl_vector_get(last_observation, 1);

	Q[x][y][act] += alpha * (reward + disc_factor * q_terminal - Q[x][y][act]);

}

void agent_cleanup() {
  /* clean up mememory */
  gsl_vector_free(local_action);
  gsl_vector_free(last_observation);
}

const char* agent_message(const char* inMessage) {
  /* might be useful to get information from the agent */
  if(strcmp(inMessage,"HELLO")==0)
  return "THIS IS SARSA";
  
  /* else */
  return "I don't know how to respond to your message";
}



/*
#ifdef EXPECTED_SARSA
static gsl_vector* local_action;
static action_t* this_action;
static gsl_vector* last_observation;

void agent_init()
{
	printf("Expected SARSA Agent\n");
  //NOT USED
}

const action_t *agent_start(const observation_t *this_observation) {
 	
  //NOT USED
  
  return this_action;
}

const action_t *agent_step(double reward, const observation_t *this_observation) {
  

  	//NOT USED
  
  	return this_action;
}


void agent_end(double reward) {
  // final learning update at end of episode
}

void agent_cleanup() {
  //clean up mememory
  gsl_vector_free(local_action);
  gsl_vector_free(last_observation);
}

const char* agent_message(const char* inMessage) {
  //might be useful to get information from the agent
  if(strcmp(inMessage,"HELLO")==0)
  return "THIS IS SARSA";
  
  // else
  return "I don't know how to respond to your message";
}
#endif

*/
