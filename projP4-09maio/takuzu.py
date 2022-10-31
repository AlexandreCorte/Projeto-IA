# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 31:
# 99048 Alexandre Corte
# 99114 Nuno Dendas

import sys
import numpy as np
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __str__(self):
        string = ""
        for i in range(0, len(self.board)):
            flatten_list = np.ravel(self.board[i])
            for j in range(0, len(self.board)-1):
                string += str(flatten_list[j])
                string += "\t"
            string += str(flatten_list[j+1])
            string +="\n"
        return string

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe

class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    def __init__(self, board,size):
        self.board = board
        self.size = size

    def get_number(self, row: int, col: int) -> int:
        return self.board[row,col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        if (row==0):
            return (None,self.board[row+1,col])
        if (row+1==self.size):
            return (self.board[row-1,col],None)
        else:
            return (self.board[row-1,col],self.board[row+1,col])

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        if (col==0):
            return (None,self.board[row,col+1])
        if (col+1==self.size):
            return (self.board[row,col-1],None)
        else:
            return (self.board[row,col-1],self.board[row,col+1])

    def double_adjacent_left(self, row:int, col:int) -> (int, int):
        if (self.size<=2):
            return (None,None)
        if (col==0 or col==1):
            return (None,None)
        else:
            return (self.board[row,col-2],self.board[row,col-1])
    
    def double_adjacent_right(self, row:int, col:int) -> (int, int):
        if (self.size<=2):
            return (None,None)
        if (col+1==self.size or col+2==self.size):
            return (None,None)
        else:
            return (self.board[row,col+1],self.board[row,col+2])

    def double_adjacent_up(self, row:int, col:int) -> (int, int):
        if (self.size<=2):
            return (None,None)
        if (row==0 or row==1):
            return (None,None)
        else:
            return (self.board[row-1,col],self.board[row-2,col])
    
    def double_adjacent_down(self, row:int, col:int) -> (int, int):
        if (self.size<=2):
            return (None,None)
        if (row+1==self.size or row+2==self.size):
            return (None,None)
        else:
            return (self.board[row+1,col],self.board[row+2,col])
 
    @staticmethod
    def parse_instance_from_stdin():
        new_data = []
        data = sys.stdin.readlines()
        number_lines = int(data[0][0])
        for i in range(1, number_lines+1):
            new_data.append(data[i].replace("\n",""))
            new_data[i-1] = new_data[i-1].split("\t")
        new_data2 = np.matrix(new_data)
        new_data2 = new_data2.astype(int)
        return (int(data[0][0]),new_data2)


    # TODO: outros metodos da classe


class Takuzu(Problem):
    def __init__(self, board: Board):
        self.initial = board
        self.size = board.size

    def actions(self, state: TakuzuState):
        board = Board(state.board, self.size)
        for i in range (0,self.size):
            flatten_list = np.ravel(state.board[i])
            count_arr_L = np.bincount(flatten_list)
            if (self.size % 2== 0):
                if(count_arr_L[1] == self.size/2 and count_arr_L[0] != self.size/2):
                    col = np.where(flatten_list == 2)
                    return [(i,col[0][0],0)]
                elif (count_arr_L[0] == self.size/2 and count_arr_L[1] != self.size/2):
                    col = np.where(flatten_list == 2)
                    return [(i,col[0][0],1)]
            else:
                if(count_arr_L[1] == int(self.size/2+1) and count_arr_L[0] != int(self.size/2-1)):
                    col = np.where(flatten_list == 2)
                    return [(i,col[0][0],0)]
                elif (count_arr_L[0] == int(self.size/2+1) and count_arr_L[1] != int(self.size/2-1)):
                    col = np.where(flatten_list == 2)
                    return [(i,col[0][0],1)]
        transp = np.transpose(state.board)
        for i in range (0,self.size):
            flatten_transp_list = np.ravel(transp[i])
            count_arr_L = np.bincount(flatten_transp_list)
            if (self.size % 2== 0):
                if(count_arr_L[1] == self.size/2 and count_arr_L[0] != self.size/2):
                    col = np.where(flatten_transp_list == 2)
                    return [(col[0][0],i,0)]
                elif (count_arr_L[0] == self.size/2 and count_arr_L[1] != self.size/2):
                    col = np.where(flatten_transp_list == 2)
                    return [(col[0][0],i,1)]
            else:
                if(count_arr_L[1] == int(self.size/2+1) and count_arr_L[0] != int(self.size/2-1)):
                    col = np.where(flatten_transp_list == 2)
                    return [(col[0][0],i,0)]
                elif (count_arr_L[0] == int(self.size/2+1) and count_arr_L[1] != int(self.size/2-1)):
                    col = np.where(flatten_transp_list == 2)
                    return [(col[0][0],i,1)]
        for i in range (0,self.size):
            flatten_list = np.ravel(self.board[i])
            for j in range (0,self.size):
                if (flatten_list[j]==2):
                    adjacent_h = np.array(board.adjacent_horizontal_numbers(i,j))
                    adjacent_v = np.array(board.adjacent_vertical_numbers(i,j))
                    if (np.all(adjacent_h==0) or np.all(adjacent_v==0)):
                        return [(i,j,1)]
                    if (np.all(adjacent_h==1) or np.all(adjacent_v==1)):
                        return [(i,j,0)]
                    double_adjacent_down = np.array(board.double_adjacent_down(i,j))
                    double_adjacent_left = np.array(board.double_adjacent_left(i,j))
                    double_adjacent_right = np.array(board.double_adjacent_right(i,j))
                    double_adjacent_up = np.array(board.double_adjacent_up(i,j))
                    if (np.all(double_adjacent_down==1) or np.all(double_adjacent_left==1)\
                        or np.all(double_adjacent_right==1) or np.all(double_adjacent_up)==1):
                        return [(i,j,0)]
                    if (np.all(double_adjacent_down==0) or np.all(double_adjacent_left==0)\
                        or np.all(double_adjacent_right==0) or np.all(double_adjacent_up)==0):
                        return [(i,j,1)]
        for i in range(0,self.size):
            random = where(np.ravel(state.board[i]) == 2)
            if(len(random)!=0):
                return [(i,random[0],0), (i,random[0],1)] 
        return

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        new_state = TakuzuState(state.board)
        new_state.board[action[0],action[1]] = action[2]
        return new_state
        

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        board = Board(state.board, self.size)
        for i in range (0,self.size):
            flatten_list = np.ravel(state.board[i])
            for j in range(0, self.size):
                if (flatten_list[j]==2):
                    return False
            count_arr_L = np.bincount(flatten_list)
            if (self.size % 2== 0):
                if (count_arr_L[0]!=count_arr_L[1]):
                    return False
            else:
                if (abs(count_arr_L[0]-count_arr_L[1])!=1):
                    return False
        transp = np.transpose(state.board)
        for i in range (0,self.size):
            flatten_list = np.ravel(transp[i])
            count_arr_L = np.bincount(flatten_list)
            if (self.size % 2== 0):
                if (count_arr_L[0]!=count_arr_L[1]):
                    return False
            else:
                if (abs(count_arr_L[0]-count_arr_L[1])!=1):
                    return False
        for i in range (0,self.size):
            flatten_list = np.ravel(state.board[i])
            for j in range (0,self.size):
                adjacent_h = np.array(board.adjacent_horizontal_numbers(i,j))
                adjacent_v = np.array(board.adjacent_vertical_numbers(i,j))
                if (np.all(adjacent_h==1) and flatten_list[j]==1):
                    return False
                if (np.all(adjacent_h==0) and flatten_list[j]==0):
                    return False
                if (np.all(adjacent_v==1) and flatten_list[j]==1):
                    return False
                if (np.all(adjacent_v==0) and flatten_list[j]==0):
                    return False
        if (len(np.unique(state.board, axis=0))!=self.size or len(np.unique(np.transpose(state.board), axis=0))!=self.size):
            return False
        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    data = Board.parse_instance_from_stdin()
    board = Board(data[1], data[0])
    problem = Takuzu(board)
    goal_node = depth_first_tree_search(problem)
    if (problem.goal_test(goal_node.state)==True):
        print(goal_node.state)
    pass
