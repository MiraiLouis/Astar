import heapq
import numpy as np

class Node:

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return (self.f, id(self)) < (other.f, id(other))


def chemin(current_node,maze):
    chemin = []
    current = current_node
    while current is not None:
        chemin.append(current.position)
        current = current.parent
    return chemin[::-1]




def modele(maze, start, end):
    maze = np.array(maze)
    if maze[end[0]][end[1]] ==1:
        return ("Destination impossible")
    else:
        map=lidar(maze, start)
        map[end[0]][end[1]]=0
        pos=start
        chem = None
        while pos!=end and chem is None:
            chem=astar(map.copy(), start, end)
            while chem is None:
                chem=astar2(map.copy(), pos, end)
                test=(0, 0)
                progress=False
                while chem !=test:
                    test=chem
                    for coord in chem:
                        pos=coord
                        temp=lidar(maze, pos)
                        prev_sum=map.sum()
                        map[temp == 0] = 0
                        if map.sum() < prev_sum:
                            progress=True
                    chem=astar(map.copy(), start, end)
                    if chem is None:
                        break
                if chem is None and not progress:
                    return "Chemin impossible"
    return astar(map, start, end)

###

def astar2(maze, start, end):
    L=[]
    if maze[end[0]][end[1]] ==1:
        print ("Destination impossible")
    else:
        lignes,colonnes = np.shape(maze)

        # Creation des nodes, initialisation de g,f et h
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialisation des listes
        open_list = []
        closed_set = set()
        heapq.heappush(open_list, start_node)


        # Boucle jusqu'à trouver end
        while open_list:

            # Node actuel
            current_node = heapq.heappop(open_list)

            # Ignorer si déjà traité
            if current_node.position in closed_set:
                continue

            # Mise à jour des listes
            closed_set.add(current_node.position)
            L.append(current_node.position)
            maze[current_node.position[0]][current_node.position[1]] = 1

            # Resultat
            if current_node == end_node:
                return chemin(current_node,maze)


            # Creation de "children"
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]: # Mouvements

                # Position actuelle
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Verifier la portée
                if node_position[0] > (lignes - 1) or node_position[0] < 0 or node_position[1] > (colonnes -1) or node_position[1] < 0:
                    continue

                # Verifier la possibilité d'y aller
                if maze[node_position[0]][node_position[1]] != 0:
                    continue

                # Creation d'un nouveau node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Boucle dans children
            for child in children:

                # Child est dans closed set
                if child.position in closed_set:
                    continue

                # Creation des f,g et h
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Append
                heapq.heappush(open_list, child)

        return L

###


def astar(maze, start, end):
    L=[]
    if maze[end[0]][end[1]] ==1:
        print ("Destination impossible")
    else:
        lignes,colonnes = np.shape(maze)

        # Creation des nodes, initialisation de g,f et h
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialisation des listes
        open_list = []
        closed_set = set()
        heapq.heappush(open_list, start_node)


        # Boucle jusqu'à trouver end
        while open_list:

            # Node actuel
            current_node = heapq.heappop(open_list)

            # Ignorer si déjà traité
            if current_node.position in closed_set:
                continue

            # Mise à jour des listes
            closed_set.add(current_node.position)
            L.append(current_node.position)
            maze[current_node.position[0]][current_node.position[1]] = 1

            # Resultat
            if current_node == end_node:
                return chemin(current_node,maze)


            # Creation de "children"
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]: # Mouvements

                # Position actuelle
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Verifier la portée
                if node_position[0] > (lignes - 1) or node_position[0] < 0 or node_position[1] > (colonnes -1) or node_position[1] < 0:
                    continue

                # Verifier la possibilité d'y aller
                if maze[node_position[0]][node_position[1]] != 0:
                    continue

                # Creation d'un nouveau node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Boucle dans children
            for child in children:

                # Child est dans closed set
                if child.position in closed_set:
                    continue

                # Creation des f,g et h
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Append
                heapq.heappush(open_list, child)


###

def lidar(maze, pos):
    map=np.ones(np.shape(maze))
    lignes,colonnes=np.shape(maze)
    x=pos[0]
    y=pos[1]
    map[x][y]=0
    for k in [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]:
        x1=x+k[0]
        y1=y+k[1]
        if 0<=x1<=lignes-1 and 0<=y1<=colonnes-1 and maze[x1][y1] == 0:
            map[x1][y1] = 0
            x2=x1+k[0]
            y2=y1+k[1]
            while 0<=x2<=lignes-1 and 0<=y2<=colonnes-1 and maze[x2][y2] == 0:
                map[x2][y2] = 0
                x2=x2+k[0]
                y2=y2+k[1]
    return(map)


maze = [
    [0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 0],
]


start = (0, 0)
end = (5, 5)

print(modele(maze,start,end))
