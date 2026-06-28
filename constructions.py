import networkx as nx
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter, deque
import pandas as pd

class GraphInstance:
    def __init__(self, n, p=0.5, has_clique=False):
        self.n = n
        self.target_clique_size = n // 2 + 1
        self.graph = nx.Graph()
        self.nodes = list(range(n))
        self.graph = nx.erdos_renyi_graph(n, p)
        
        if has_clique:
            self._inject_clique(self.target_clique_size)
        else:
            self._ensure_no_large_clique(self.target_clique_size)
        self._compute_global_complement_matching()

    def _inject_clique(self, tam_clique):
        # Selecciona n/2 + 1 nodos al azar y los conecta todos entre sí
        clique_nodes = random.sample(self.nodes, tam_clique)
        for u, v in itertools.combinations(clique_nodes, 2):
            self.graph.add_edge(u, v)
        self.clique_nodes = set(clique_nodes)

    def _ensure_no_large_clique(self,tam_clique):
        while True:
            clique = nx.algorithms.clique.max_weight_clique(self.graph, weight=None)
            if len(clique[0]) < tam_clique:
                break
            # Romper el clique eliminando una arista
            u, v = random.sample(clique[0], 2)
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)

    def get_neighbors(self, node): #dado un nodo del grafo nos devuelve una lista con sus vecinos y él mismo 
        return set(self.graph.neighbors(node)) | {node}
        
    def _compute_global_complement_matching(self):
        """
        Calcula el matching máximo en el grafo complementario usando Edmonds.
        Guarda las parejas sin aristas para penalizaciones rápidas en los tests.
        """
        # 1. Crear el grafo complementario (aristas que NO existen en el original)
        G_comp = nx.complement(self.graph)
        
        # 2. Aplicar Edmonds (Matching de cardinalidad máxima)
        matching = nx.max_weight_matching(G_comp, maxcardinality=True)
        
        # 3. Guardar resultados como atributos de la instancia
        self.max_matching_size = len(matching)
        self.max_matching_pairs = list(matching)
        
        # 4. Caché de acceso rápido O(1) para saber si un nodo está emparejado
        self.matched_nodes = set()
        for u, v in self.max_matching_pairs:
            self.matched_nodes.add(u)
            self.matched_nodes.add(v)
            
    def get_matching_pairs_in_subset(self, subset):
        """
        Filtra las parejas globales pre-calculadas para devolver solo 
        aquellas donde AMBOS nodos pertenecen al subconjunto dado.
        """
        subset_set = set(subset)
        return [(u, v) for u, v in self.max_matching_pairs if u in subset_set and v in subset_set]
    
class SpiderWeb:
    def __init__(self, n_nodes, seed=None):
        assert n_nodes % 2 == 0, "n debe ser par"
        if seed is not None:
            random.seed(seed)
        self.n = n_nodes
        self.nodes = list(range(n_nodes))
 
        self.global_partitions = []   # lista de (frozenset, frozenset): los bits del certificado
        self.group_definitions = []   # cada grupo: piezas, ids de sus 4 bits y sus 4 particiones
        self._partition_ids = {}      # dedup de 2-particiones -> id de bit
        self._group_keys = {}         # dedup de grupos -> indice
        self.adj = {}                 # indice_grupo -> set de indices_grupo que comparten >=1 bit
        self._share = Counter()       # id de bit -> en cuantos grupos aparece (incremental)
 
        self._crear_grupo_base()
 
 
    def _id_particion(self, L, R):
        L, R = frozenset(L), frozenset(R)
        clave = frozenset((L, R))
        if clave in self._partition_ids:
            return self._partition_ids[clave]
        self.global_partitions.append((L, R))
        bit = len(self.global_partitions) - 1
        self._partition_ids[clave] = bit
        return bit
 
    @staticmethod
    def _clave_grupo(A, B, C, Ap, Bp, Cp):
        # identifica el grupo por sus 3 pares de piezas, sin importar orden ni cual es "primado"
        return frozenset(frozenset((frozenset(u), frozenset(v)))
                         for (u, v) in [(A, Ap), (B, Bp), (C, Cp)])
 
    def _crear_grupo(self, A, B, C, Ap, Bp, Cp):
        """Crea (o recupera por dedup) el grupo. Devuelve su indice y conecta aristas."""
        assert len(A) == len(Ap) and len(B) == len(Bp) and len(C) == len(Cp), \
            "las piezas deben cumplir |A|=|A'|, |B|=|B'|, |C|=|C'|"
 
        clave = self._clave_grupo(A, B, C, Ap, Bp, Cp)
        if clave in self._group_keys:
            return self._group_keys[clave]          # ya existia -> punto de convergencia
 
        X, Xp = A | B | C, Ap | Bp | Cp
        Y, Yp = A | B | Cp, Ap | Bp | C
        Z, Zp = A | Bp | C, Ap | B | Cp
        Wv, Wp = Ap | B | C, A | Bp | Cp
        ids = (self._id_particion(X, Xp), self._id_particion(Y, Yp),
               self._id_particion(Z, Zp), self._id_particion(Wv, Wp))
 
        idx = len(self.group_definitions)
        self.group_definitions.append({
            'pieces': (set(A), set(B), set(C), set(Ap), set(Bp), set(Cp)),
            'ids': ids,
            'partitions': [(X, Xp), (Y, Yp), (Z, Zp), (Wv, Wp)],
        })
        self._group_keys[clave] = idx
        self.adj[idx] = set()
 
        bits = set(ids)
        for j, g in enumerate(self.group_definitions[:-1]):
            if bits & set(g['ids']):
                self.adj[idx].add(j)
                self.adj[j].add(idx)
        for b in bits:
            self._share[b] += 1
        return idx
 
    def _crear_grupo_base(self):
        s = random.sample(self.nodes, self.n)
        h = self.n // 2
        sA = max(1, h // 3)
        sB = max(1, (h - sA) // 2)
        sC = h - sA - sB
        assert sC >= 1, "n demasiado pequeno para tres piezas no vacias (usa n >= 6)"
        A = set(s[0:sA]);          B = set(s[sA:sA + sB]);          C = set(s[sA + sB:h])
        Ap = set(s[h:h + sA]);     Bp = set(s[h + sA:h + sA + sB]); Cp = set(s[h + sA + sB:])
        self._crear_grupo(A, B, C, Ap, Bp, Cp)
 
 
    @staticmethod
    def _piezas_de_particion(g, cual):
        """Pares (Li, Ri) (Li izquierda, Ri derecha) de la particion cual in {0:X,1:Y,2:Z,3:W}."""
        A, B, C, Ap, Bp, Cp = g['pieces']
        return {
            0: [(A, Ap), (B, Bp), (C, Cp)],   # X
            1: [(A, Ap), (B, Bp), (Cp, C)],   # Y
            2: [(A, Ap), (Bp, B), (C, Cp)],   # Z
            3: [(Ap, A), (B, Bp), (C, Cp)],   # W
        }[cual]
 
    @staticmethod
    def _resplit(S, k1):
        L = list(S)
        random.shuffle(L)
        return set(L[:k1]), set(L[k1:])
 
    def _reinterpretar(self, pares, modo):
        """
        De los 3 pares (Li,Ri) de UNA 2-particion produce 3 pares NUEVOS que describen
        la MISMA 2-particion con otra estructura de piezas.
          '2.1': se conserva un par y se re-divide la union de los otros dos (2 -> 2).
          '2.2': se fusionan dos pares en uno y el tercero se parte en dos (2 -> 1 y 1 -> 2).
        Devuelve (A,B,C,Ap,Bp,Cp) o None si no es viable.
        """
        (L0, R0), (L1, R1), (L2, R2) = pares
        if modo == '2.1':
            keep = random.randrange(3)
            i, j = [k for k in range(3) if k != keep]
            Ls = [L0, L1, L2]; Rs = [R0, R1, R2]
            ni = len(Ls[i])
            Ls[i], Ls[j] = self._resplit(Ls[i] | Ls[j], ni)
            Rs[i], Rs[j] = self._resplit(Rs[i] | Rs[j], ni)   # |Ri|=|Li| -> tamanos casan
            return (Ls[0], Ls[1], Ls[2], Rs[0], Rs[1], Rs[2])
        # modo '2.2'
        cand = [k for k in range(3) if len(pares[k][0]) >= 2]
        if not cand:
            return None
        s = random.choice(cand)
        j, k = [t for t in range(3) if t != s]
        Lf, Rf = pares[j][0] | pares[k][0], pares[j][1] | pares[k][1]   # fusion
        La, Lb = self._resplit(pares[s][0], 1)
        Ra, Rb = self._resplit(pares[s][1], 1)
        return (La, Lb, Lf, Ra, Rb, Rf)
 
 
    def divergir(self, idx_grupo=None, cual=None, modo=None, reparto='uniforme'):
        """
        Toma un grupo, elige UNA de sus 4 particiones, la reinterpreta y crea un grupo
        NUEVO que comparte EXACTAMENTE ese bit con el de origen (arista del arbol).
 
        reparto='aleatorio' : elige la particion al azar (fiel a "una cualquiera de las 4").
        reparto='uniforme'  : elige la particion cuyo bit este MENOS repartido, para que
                              ningun bit se convierta en un cuello de botella.
        """
        if idx_grupo is None:
            idx_grupo = random.randrange(len(self.group_definitions))
        g = self.group_definitions[idx_grupo]
 
        if cual is None:
            if reparto == 'uniforme':
                cual = min(range(4), key=lambda c: self._share[g['ids'][c]])
            else:
                cual = random.randrange(4)
        if modo is None:
            modo = random.choice(['2.1', '2.2'])
 
        pares = self._piezas_de_particion(g, cual)
        nuevo = self._reinterpretar(pares, modo) or self._reinterpretar(pares, '2.1')
        return self._crear_grupo(*nuevo)
 
 
    def converger(self, g1, g2):
        """
        Construye el HIJO COMUN de g1 y g2: el unico grupo que contiene una particion P
        de g1 (como su X) y una particion Q de g2 (como su W). Asi el grupo nuevo comparte
        un bit con g1 y otro distinto con g2, cerrando un ciclo entre dos ramas.
 
        Refinamiento de P=(P1,P2) y Q=(R1,R2):
          acuerdo  P1&R1 -> A ,  P2&R2 -> A'
          desacuerdo P1&R2 -> se parte en (B,C) ,  P2&R1 -> se parte en (B',C')
        Como P y Q son 2-particiones balanceadas, |A|=|A'| y |B,C| casan con |B',C'|
        automaticamente. Devuelve el indice del hijo comun, o None si no es posible.
        """
        for (P1, P2) in self.group_definitions[g1]['partitions']:
            for (Q1, Q2) in self.group_definitions[g2]['partitions']:
                for (R1, R2) in [(Q1, Q2), (Q2, Q1)]:        # dos orientaciones de Q
                    A = P1 & R1; Ap = P2 & R2
                    d1 = P1 & R2; d2 = P2 & R1
                    if A and Ap and len(d1) >= 2 and len(d2) >= 2:
                        B, C = self._resplit(d1, 1)
                        Bp, Cp = self._resplit(d2, 1)
                        return self._crear_grupo(A, B, C, Ap, Bp, Cp)
        return None
 
 
    def construir(self, n_divergencias=20, converger_cada=4,
                  intentos_conv=12, reparto='uniforme'):
        """Alterna divergencias (crecimiento en arbol) con convergencias (cierre de ciclos)."""
        for paso in range(1, n_divergencias + 1):
            self.divergir(reparto=reparto)
            if converger_cada and paso % converger_cada == 0 and len(self.group_definitions) >= 2:
                for _ in range(intentos_conv):
                    g1, g2 = random.sample(range(len(self.group_definitions)), 2)
                    antes = len(self.group_definitions)
                    if self.converger(g1, g2) is not None and len(self.group_definitions) > antes:
                        break    # se anadio un grupo nuevo que cierra un ciclo
        return self    def __init__(self, n_nodes):
        self.n = n_nodes
        self.nodes = list(range(n_nodes))
        self.tam = 4
        self.global_partitions = []
        self.group_definitions = []
        # --- anadido: dedup + grafo (para que la telarana conecte y converja) ---
        self._partition_ids = {}   # clave-conjunto -> id (dedup de 2-particiones)
        self._group_ids = {}       # clave-grupo -> indice (dedup de grupos)
        self.adj = {}              # indice_grupo -> set((otro, id_particion_compartida))
        self.convergences = []     # indices de grupo alcanzados mas de una vez
        # Iniciar la telaraña con el Grupo 0 (Paso 1)
        self._initialize_base_group()

    def _create_partition_id(self, P, P_prime):
        # dedup: si la 2-particion ya existe (mismo contenido), devuelve su id
        key = frozenset((frozenset(P), frozenset(P_prime)))
        if key in self._partition_ids:
            return self._partition_ids[key]
        self.global_partitions.append((frozenset(P), frozenset(P_prime)))
        pid = len(self.global_partitions) - 1
        self._partition_ids[key] = pid
        return pid

    #Dados 3 conjuntos, crea un grupo de 4 2-particiones
    def _create_group_from_pieces(self, A, B, C, Ap, Bp, Cp):
        X = A | B | C;        Xp = Ap | Bp | Cp
        Y = A | B | Cp;       Yp = Ap | Bp | C
        Z = A | Bp | C;       Zp = Ap | B | Cp
        W = Ap | B | C;       Wp = A | B | Cp
        
        # dedup de grupo: la terna de pares de piezas identifica el grupo
        # (sin importar orden ni cual es 'primado'). Si ya existe -> convergencia.
        gkey = frozenset(frozenset((frozenset(u), frozenset(v)))
                         for (u, v) in [(A, Ap), (B, Bp), (C, Cp)])
        if gkey in self._group_ids:
            self.convergences.append(self._group_ids[gkey])
            return X, Xp
        
        id_X = self._create_partition_id(X, Xp)
        id_Y = self._create_partition_id(Y, Yp)
        id_Z = self._create_partition_id(Z, Zp)
        id_W = self._create_partition_id(W, Wp)
        
        # Guardamos el detalle para que el verificador pueda leerlo
        self.group_definitions.append({
            'partitions': [(X,Xp), (Y,Yp), (Z,Zp), (W,Wp)],
            'ids': (id_X, id_Y, id_Z, id_W),
            'pieces': (set(A), set(B), set(C), set(Ap), set(Bp), set(Cp))  # para divergir desde aqui
        })
        
        # --- anadido: registrar el grupo y conectar con los que comparten particion ---
        idx = len(self.group_definitions) - 1
        self._group_ids[gkey] = idx
        self.adj[idx] = set()
        pid_set = {id_X, id_Y, id_Z, id_W}
        for j, g in enumerate(self.group_definitions[:-1]):
            for sp in (pid_set & set(g['ids'])):
                self.adj[idx].add((j, sp)); self.adj[j].add((idx, sp))
        
        return X, Xp # Devolvemos la partición base por si queremos expandir desde ella

    def _initialize_base_group(self): 
        """Paso 1: Generamos el primer grupo de 4 particiones de la nada"""
        shuffled = random.sample(self.nodes, self.n)
        half = self.n // 2
        
        # Repartimos tamaños (Para N=8, sizes serán 1, 1, 2)
        sA = half // 3
        sB = (half - sA) // 2
        sC = half - sA - sB
        
        self.A = set(shuffled[0 : sA])
        self.B = set(shuffled[sA : sA+sB])
        self.C = set(shuffled[sA+sB : half])
        
        self.Ap = set(shuffled[half : half+sA])
        self.Bp = set(shuffled[half+sA : half+sA+sB])
        self.Cp = set(shuffled[half+sA+sB : ])
        
        self._create_group_from_pieces(self.A, self.B, self.C, self.Ap, self.Bp, self.Cp)

    def _split_set(self, S, size1, size2):
        """Función auxiliar para cortar un conjunto en dos tamaños específicos"""
        L = list(S)
        random.shuffle(L)
        return set(L[:size1]), set(L[size1:])

    def expand_web(self, steps=5, converge_every=5):
        """
        Pasos 2 y 3: Expande la telaraña divergiendo desde una FRONTERA de nodos
        (no una cadena). En 2.1/2.2 va rotando sobre qué piezas opera.
        """
        nuevo_tam = self.tam+steps*4
        if not hasattr(self, 'frontier'):
            self.frontier = list(range(len(self.group_definitions)))  # arranca con el grupo base
        if not hasattr(self, '_since_conv'):
            self._since_conv = 0
            
        for i in range(steps):
            # Elegimos un nodo CUALQUIERA de la frontera y leemos sus piezas
            idx = random.choice(self.frontier)
            A, B, C, Ap, Bp, Cp = self.group_definitions[idx]['pieces']
            U = [set(A), set(B), set(C)]      # lados no primados
            P = [set(Ap), set(Bp), set(Cp)]   # lados primados

            # 2.1: dejamos UNA pieza fija (rotando cuál) y re-dividimos las otras dos
            keep = random.randrange(3)
            i1, i2 = [k for k in range(3) if k != keep]
            U[i1], U[i2] = self._split_set(U[i1] | U[i2], len(U[i1]), len(U[i2]))
            P[i1], P[i2] = self._split_set(P[i1] | P[i2], len(P[i1]), len(P[i2]))

             # Paso 3: construimos el grupo y, si es nuevo, lo añadimos a la frontera
            before = len(self.group_definitions)
            self._create_group_from_pieces(U[0], U[1], U[2], P[0], P[1], P[2])
            if len(self.group_definitions) > before:
                self.frontier.append(len(self.group_definitions) - 1)
                self._since_conv += 1
            self.tam+=4
            if self.tam == nuevo_tam:
                break
            # 2.2: fusionamos DOS piezas en una y partimos la TERCERA en dos (rotando cuál)
            candidates = [k for k in range(3) if len(U[k]) >= 2 and len(P[k]) >= 2]
            if not candidates:
                continue
            s = random.choice(candidates)
            j1, j2 = [k for k in range(3) if k != s]
            fused_U = U[j1] | U[j2];   fused_P = P[j1] | P[j2]
            a_U, b_U = self._split_set(U[s], 1, len(U[s]) - 1)
            a_P, b_P = self._split_set(P[s], 1, len(P[s]) - 1)
            U = [a_U, b_U, fused_U]
            P = [a_P, b_P, fused_P]
            
            # Paso 3: construimos el grupo y, si es nuevo, lo añadimos a la frontera
            before2 = len(self.group_definitions)
            self._create_group_from_pieces(U[0], U[1], U[2], P[0], P[1], P[2])
            if len(self.group_definitions) > before2:
                self.frontier.append(len(self.group_definitions) - 1)
                self._since_conv += 1
            if len(self.group_definitions) > before:
                self.frontier.remove(idx)
            self.tam+=4
            if self.tam == nuevo_tam:
                break
            # Paso 4: cada 'converge_every' grupos nuevos, tejemos un hijo común
            if self._since_conv >= converge_every and len(self.frontier) >= 2:
                g1, g2 = random.sample(self.frontier, 2)
                self.force_convergence(g1, g2)
                self._since_conv = 0
            self.tam+=4
            if self.tam == nuevo_tam:
                break

    def force_convergence(self, g1, g2):
        """
        Paso 4: construye el HIJO COMUN de g1 y g2.
        Combina UNA 2-particion P de g1 con OTRA Q de g2 (distintas) y crea el
        grupo que las contiene a ambas: así g3 es hijo de g1 (reinterpretando P)
        y de g2 (reinterpretando Q) a la vez. NO es re-trocear una particion ya
        compartida. Devuelve el indice del hijo comun, o None si no es posible.
 
        Refinamiento de P=(P1,P2) y Q=(R1,R2):
          - celdas de ACUERDO  -> pareja que se conserva (A, A')
          - celdas de DESACUERDO -> se parten en (B,C) y (B',C')
        """
        def _gkey(A, B, C, Ap, Bp, Cp):
            return frozenset(frozenset((frozenset(u), frozenset(v)))
                             for (u, v) in [(A, Ap), (B, Bp), (C, Cp)])
 
        for (P1, P2) in self.group_definitions[g1]['partitions']:
            for (Q1, Q2) in self.group_definitions[g2]['partitions']:
                for (R1, R2) in [(Q1, Q2), (Q2, Q1)]:   # dos orientaciones de Q
                    A  = P1 & R1     # acuerdo lado1  -> A
                    Ap = P2 & R2     # acuerdo lado2  -> A'
                    d1 = P1 & R2     # desacuerdo lado1 -> se parte en B, C
                    d2 = P2 & R1     # desacuerdo lado2 -> se parte en B', C'
                    if A and Ap and len(d1) >= 2 and len(d2) >= 2:
                        B, C   = self._split_set(d1, 1, len(d1) - 1)
                        Bp, Cp = self._split_set(d2, 1, len(d2) - 1)
                        gk = _gkey(A, B, C, Ap, Bp, Cp)
                        self._create_group_from_pieces(A, B, C, Ap, Bp, Cp)
                        g3 = self._group_ids[gk]
                        if hasattr(self, 'frontier') and g3 not in self.frontier:
                            self.frontier.append(g3)
                        return g3
        return None
    
class PureGeneticAlgorithm:
    def __init__(self, num_bits, fitness_evaluator, pop_size=100, generations=50, mut_rate=0.15, seed_individual=None):
        """
        num_bits: Longitud del cromosoma (tamaño del certificado).
        fitness_evaluator: Una función que recibe un certificado y devuelve un float [0.0, 1.0].
        """
        self.num_bits = num_bits
        self.fitness_evaluator = fitness_evaluator
        self.pop_size = pop_size
        self.generations = generations
        self.mut_rate = mut_rate
        self.seed = seed_individual
        self.elite_size=5

    def _create_individual(self):
        return [random.choice([0, 1]) for _ in range(self.num_bits)]

    def _inicializar_poblacion(self, n_individuos):
        poblacion = []
        if self.seed == None:
            for _ in range(self.pop_size):
                individuo = self._create_individual()
                #individuo = self.generar_individuo_2k(num_grupos=len(web.groups), k=16, bits_per_c=5)
                poblacion.append(individuo)
            return poblacion
            
        # 1. Un individuo con el certificado greedy
        poblacion.append(self.seed)
        
        # 2. Variantes mutadas del greedy
        for _ in range(n_individuos // 2):
            mutado = self._mutate(self.seed)
            poblacion.append(mutado)
        
        # 3. El resto aleatorio para mantener diversidad
        while len(poblacion) < n_individuos:
            poblacion.append(self._create_individual())
        
        return poblacion
    
    def _crossover(self, parent1, parent2):
        punto = random.randrange(1, self.num_bits)
        return parent1[:punto] + parent2[punto:]
    
    """    
    def _mutate(self, individual):
        bit = random.randrange(0,self.num_bits)
        if random.random() < self.mut_rate:
            individual[bit]=1-individual[bit]
        return individual
    """
    def _mutate(self, individual):
        return [1 - b if random.random() < self.mut_rate else b for b in individual]
       
    def _torneo(self, population, fitness_scores, k=3):
        """Selección por torneo: elige k candidatos al azar y devuelve el mejor."""
        candidatos = random.sample(range(len(population)), k)
        mejor = max(candidatos, key=lambda i: fitness_scores[i])
        return population[mejor]
        
    def run(self):
        #population = [self._create_individual() for _ in range(self.pop_size)]
        # Ejemplo para crear una población inicial de 100 individuos
        population = self._inicializar_poblacion(self.pop_size)          
        best_fitness = -1.0
        best_individual = None
        
        # --- OPTIMIZACIÓN 3: El Diccionario Caché ---
        fitness_cache = {}
        
        def get_fitness(ind):
            # Convertimos la lista a tupla para poder usarla de llave en el diccionario
            tupla_ind = tuple(ind)
            if tupla_ind not in fitness_cache:
                fitness_cache[tupla_ind] = self.fitness_evaluator(ind)
            return fitness_cache[tupla_ind]

        for generation in range(self.generations):
            # Usamos la función con caché en lugar de llamar directamente al evaluador
            fitness_scores = [get_fitness(ind) for ind in population]
            
            current_best_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx][:]
                
            if best_fitness == 1.0:
                break 
                
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            new_population = [population[i][:] for i in elite_indices] 
            while len(new_population) < self.pop_size:
                p1 = self._torneo(population, fitness_scores)
                p2 = self._torneo(population, fitness_scores)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)
                
            population = new_population
            
        return best_fitness, best_individual
    

    aristas_sorted = sorted(total_dict.keys())
    pct_inverso, pct_pcp, pct_fp, avg_fitness = [], [], [], []
    
    # Procesamos los datos para cada densidad de aristas
    for m in aristas_sorted:
        total = total_dict[m]
        pct_inverso.append((inv_dict[m] / total) * 100)
        pct_pcp.append((pcp_dict[m] / total) * 100)
        pct_fp.append((fp_dict[m] / total) * 100)
        
        # CÁLCULO DEL FITNESS MEDIO: (Fitness total acumulado / número de grafos evaluados)
        media_fit = (fitness_dict[m] / total) * 100
        avg_fitness.append(media_fit)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    peor_fitness_sorted = [peor_fitness[m] for m in aristas_sorted]
    if clique:
        # Instancias positivas
        label_inverso  = 'Filtro Edmonds'
        label_pcp      = 'AG no encontró certificado válido'
        label_fp       = 'AG encontró certificado válido'
        color_inverso  = 'lightblue'
        color_pcp      = 'salmon'
        color_fp       = 'green'
        titulo         = f"Instancias Positivas ({test_name})"
        titulo_fitness = "Fitness Medio — Instancias Positivas (Con Clique)"
        titulo_peor_fitness = "Fitness Mejor Adversario — Instancias Positivas (Con Clique)"
    else:
        # Instancias negativas
        label_inverso  = 'Filtro Edmonds'
        label_pcp      = 'Cazados por Test PCP'
        label_fp       = 'Falsos Positivos (Mentiroso Perfecto)'
        color_inverso  = 'lightblue'
        color_pcp      = 'steelblue'
        color_fp       = 'red'
        titulo         = f"Instancias Negativas ({test_name})"
        titulo_fitness = "Fitness Medio — Instancias Negativas (Sin Clique)"
        titulo_peor_fitness = "Fitness Mejor Adversario — Instancias Negativas (Sin Clique)"

    # Gráfica 1: Barras apiladas
    ax1.bar(aristas_sorted, pct_inverso, color=color_inverso, label=label_inverso)
    ax1.bar(aristas_sorted, pct_pcp, bottom=pct_inverso, color=color_pcp, label=label_pcp)
    bottom_fp = [i + j for i, j in zip(pct_inverso, pct_pcp)]
    ax1.bar(aristas_sorted, pct_fp, bottom=bottom_fp, color=color_fp, label=label_fp)
    ax1.set_title(titulo)
    ax1.set_xlabel("Número de Nodos (n)")
    ax1.set_ylabel("Porcentaje de Casos (%)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Gráfica 2: peor fitness
    color_fitness = 'green' if clique else 'purple'
    ax2.plot(aristas_sorted, peor_fitness_sorted, marker='o', color=color_fitness, linestyle='-', linewidth=2)
    ax2.fill_between(aristas_sorted, peor_fitness_sorted, color=color_fitness, alpha=0.1)
    ax2.set_title(titulo_peor_fitness)
    ax2.set_xlabel("Número de Nodos (n)")
    ax2.set_ylabel("Fitness Mayor Adversario (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 