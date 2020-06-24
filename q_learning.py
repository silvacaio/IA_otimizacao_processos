# Otimizando fluxos de armazém com o Q-Learning

# Importação das bibliotecas
import numpy as np

# Configuração de gamma e alpha
from numpy.core._multiarray_umath import ndarray

gamma = 0.75
alpha = 0.9  # taxa de aprendizagem

# PARTE 1 - DEFINIÇÃO DO AMBIENTE
location_to_state = {"A": 0,
                     "B": 1,
                     "C": 2,
                     "D": 3,
                     "E": 4,
                     "F": 5,
                     "G": 6,
                     "H": 7,
                     "I": 8,
                     "J": 9,
                     "K": 10,
                     "L": 11}

actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# maiores valores são as maiores recompensas, devido as prioridades


R = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])

# PARTE 3 - RESULTADOS (ENTRANDO EM PRODUÇÃO)

state_to_location = {state: location for location, state in
                     location_to_state.items()}  # invertendo o dicionario, as chaves viram valores e os valores viram as chaves


# PARTE 2 - CONSTUÇÃO DA SOLUÇÃO DE IA COM Q-LEARNING

def training_data_set(r_values):
    Q = np.array(np.zeros([12, 12]))  # recompensas

    for i in range(1000):
        current_state = np.random.randint(0, 12)  # estado aleatorio para começar interação
        playable_actions = []  # ações possiveis daquele estado
        for j in range(12):
            if r_values[current_state, j] > 0:
                playable_actions.append(j)

        next_state = np.random.choice(playable_actions)  # próximo estado escolhido aleatoriamente
        TD = r_values[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[
            current_state, next_state]  # calculo da diferença temporal

        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD  # atualização na tabela de recompensas

    return Q


def route(starting_location, ending_location):
    R_new = np.copy(R)  # copia para mudar o prioridade

    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000

    Q = training_data_set(R_new)

    route_choose = [starting_location]  # primeiro item do array é a posição inicial
    next_location = starting_location

    while next_location != ending_location:
        starting_state = location_to_state[starting_location]
        next_state_choose = np.argmax(Q[starting_state])
        next_location = state_to_location[next_state_choose]
        route_choose.append(next_location)
        starting_location = next_location

    return route_choose


def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]


# impressão da rota final
print(best_route("E", "K", "G"))
