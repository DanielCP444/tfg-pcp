import itertools
import random
import math
import networkx as nx

class test_t2grupos_solovencedor:
    def __init__(self, graph_instance, web, t=4, max_tests=20):
        """
        t2: Número de GRUPOS que el verificador seleccionará en cada test.
        El número real de bits leídos será t = t2 * 4.
        """
        self.G = graph_instance
        self.t2 = t // 4
        self.grupos = web.group_definitions if hasattr(web, 'group_definitions') else web.groups
        self.num_grupos = len(self.grupos)
        self.num_bits = self.num_grupos * 4 # Certificado lineal
        
        # --- CACHÉS PARA VELOCIDAD MÁXIMA ---
        self.neighbors_cache = {u: set(self.G.graph.neighbors(u)) | {u} for u in range(self.G.n)}
        self.threshold = self.G.n // 4 + 1 
        
        # Mapeamos a qué partición matemática real pertenece cada bit absoluto (0 a num_bits-1)
        self.partition_ids = []
        unique_partitions = {}
        
        for g in self.grupos:
            for p, p_prime in g['partitions']:
                norm_p, norm_p_prime = (p, p_prime) if 0 in p else (p_prime, p)
                key = (frozenset(norm_p), frozenset(norm_p_prime))
                
                if key not in unique_partitions:
                    unique_partitions[key] = len(unique_partitions)
                self.partition_ids.append(unique_partitions[key])
                
        # Pre-calcular matchings para todas las particiones posibles
        self._matching_cache = {}
        for grupo in self.grupos:
            for P, P_prime in grupo['partitions']:
                for subset in [P, P_prime]:
                    key = frozenset(subset)
                    if key not in self._matching_cache:
                        S = list(subset)
                        G_comp = nx.Graph()
                        G_comp.add_nodes_from(S)
                        for idx1 in range(len(S)):
                            for idx2 in range(idx1 + 1, len(S)):
                                u, v = S[idx1], S[idx2]
                                if v not in self.neighbors_cache[u]:
                                    G_comp.add_edge(u, v)
                        matching = nx.max_weight_matching(G_comp, maxcardinality=True)
                        matched = set(n for pair in matching for n in pair)
                        free = [n for n in S if n not in matched]
                        self._matching_cache[key] = (matching, free)

        # --- GENERACIÓN DE TESTS (Por Grupos) ---
        total_posibles = math.comb(self.num_grupos, self.t2)
        
        if total_posibles <= max_tests:
            self.tests_a_evaluar = list(itertools.combinations(range(self.num_grupos), self.t2))
        else:
            tests_set = set()
            while len(tests_set) < max_tests:
                nuevo_test = tuple(sorted(random.sample(range(self.num_grupos), self.t2)))
                tests_set.add(nuevo_test)
            self.tests_a_evaluar = list(tests_set)
            
        self.total_tests = len(self.tests_a_evaluar)
        self.forbidden = [(1, 0, 0, 0), (0, 1, 1, 1)]

    def _evaluate_test(self, group_indices, cert):
        """
        group_indices: tupla con los índices de los grupos elegidos (ej: (1, 3))
        """
        bits_evaluar = []
        particiones_evaluar = []
        ids_absolutos = []
        
        # 1. Recopilamos los t2 * 4 bits de los grupos seleccionados
        for g_idx in group_indices:
            grupo = self.grupos[g_idx]
            bits_del_grupo = []
            for i in range(4):
                bit_idx = g_idx * 4 + i
                ids_absolutos.append(bit_idx)
                bit_val = cert[bit_idx]
                bits_evaluar.append(bit_val)
                bits_del_grupo.append(bit_val)
                particiones_evaluar.append(grupo['partitions'][i])
            if tuple(bits_del_grupo) in self.forbidden:
                return 0
                
        total_bits_leidos = self.t2 * 4
        
        # 3. REGLA 2: TOPOLOGÍA (Todos contra todos en los grupos elegidos)
        selected_subsets = [P if bit == 0 else P_prime for bit, (P, P_prime) in zip(bits_evaluar, particiones_evaluar)]

        total_cajas = len(selected_subsets) 
        matchings_por_caja = []
        free_nodes_por_caja = []
        
        for j in range(total_cajas):
            key = frozenset(selected_subsets[j])
            matching_j, free_j = self._matching_cache[key]
            matchings_por_caja.append(matching_j)
            free_nodes_por_caja.append(free_j)
        
        for i,Ai in enumerate(selected_subsets):
            valid_nodes = set()
            valid_nodes_in_Ai = 0
            for node_u in Ai:
                neighbors = self.neighbors_cache[node_u]
                if all(len(neighbors.intersection(Aj)) >= self.threshold for Aj in selected_subsets):
                    valid_nodes_in_Ai += 1
                    valid_nodes.add(node_u)
            if valid_nodes_in_Ai < self.threshold: 
                return 0 # ¡Cazado! No hay nodos suficientes para formar el clique.
                
            # PASO B: Conteo Efectivo Final en la caja i
            effective_valid_count = 0
            
            # 1. Nodos libres que sobrevivieron
            for free_node in free_nodes_por_caja[i]:
                if free_node in valid_nodes:
                    effective_valid_count += 1
                    
            # 2. Parejas congeladas donde AL MENOS UNO sobrevivió
            for v1, v2 in matchings_por_caja[i]:
                if v1 in valid_nodes or v2 in valid_nodes:
                    effective_valid_count += 1
                    
            if effective_valid_count < self.threshold:
                return 0   
        return 1

    def calc_fitness(self, cert):
        if self.total_tests == 0: return 0.0
        passed_tests = sum(self._evaluate_test(test, cert) for test in self.tests_a_evaluar)
        return passed_tests / self.total_tests

class test_t2grupos_diferenciaexacta:
    def __init__(self, graph_instance, web, t=4, max_tests=20):
        """
        t2: Número de GRUPOS que el verificador seleccionará en cada test.
        """
        self.G = graph_instance
        self.t2 = t // 4
        self.grupos = web.group_definitions if hasattr(web, 'group_definitions') else web.groups
        self.num_grupos = len(self.grupos)
        self.k = self.G.target_clique_size
        
        # Rango de nodos válidos (c)
        self.valid_c_values = list(range(self.k + 1))
        self.bits_per_c = math.ceil(math.log2(len(self.valid_c_values)))
        self.num_bits = self.num_grupos * 4 * self.bits_per_c 
        
        # --- CACHÉS ---
        self.neighbors_cache = {u: set(self.G.graph.neighbors(u)) | {u} for u in range(self.G.n)}
        
        # Mapeamos a qué partición matemática real pertenece cada grupo lógico de bits
        self.partition_ids = []
        unique_partitions = {}
        for g in self.grupos:
            for p, p_prime in g['partitions']:
                key = (frozenset(p), frozenset(p_prime))
                if key not in unique_partitions:
                    unique_partitions[key] = len(unique_partitions)
                self.partition_ids.append(unique_partitions[key])

        # Pre-calcular matchings para todas las particiones posibles
        self._matching_cache = {}
        for grupo in self.grupos:
            for P, P_prime in grupo['partitions']:
                for subset in [P, P_prime]:
                    key = frozenset(subset)
                    if key not in self._matching_cache:
                        S = list(subset)
                        G_comp = nx.Graph()
                        G_comp.add_nodes_from(S)
                        for idx1 in range(len(S)):
                            for idx2 in range(idx1 + 1, len(S)):
                                u, v = S[idx1], S[idx2]
                                if v not in self.neighbors_cache[u]:
                                    G_comp.add_edge(u, v)
                        matching = nx.max_weight_matching(G_comp, maxcardinality=True)
                        matched = set(n for pair in matching for n in pair)
                        free = [n for n in S if n not in matched]
                        self._matching_cache[key] = (matching, free)

        # --- GENERACIÓN DE TESTS (Por Grupos) ---
        total_posibles = math.comb(self.num_grupos, self.t2)
        if total_posibles <= max_tests:
            self.tests_a_evaluar = list(itertools.combinations(range(self.num_grupos), self.t2))
        else:
            tests_set = set()
            while len(tests_set) < max_tests:
                tests_set.add(tuple(sorted(random.sample(range(self.num_grupos), self.t2))))
            self.tests_a_evaluar = list(tests_set)
            
        self.total_tests = len(self.tests_a_evaluar)
        self.cert_greedy = self._construir_certificado_greedy()
        
    def _construir_certificado_greedy(self):
        cert = []
        for grupo in self.grupos:
            c_del_grupo = []
            
            for P, P_prime in grupo['partitions']:
                # Estimar c proporcionalmente al grado medio
                grado_P = sum(self.G.graph.degree(u) for u in P) / len(P)
                grado_Pp = sum(self.G.graph.degree(u) for u in P_prime) / len(P_prime)
                
                total_grado = grado_P + grado_Pp
                if total_grado == 0:
                    c = self.k // 2
                else:
                    c = round(self.k * grado_P / total_grado)
                
                c = max(0, min(c, self.k))
                c_del_grupo.append(c)
            
            # Ajustar para que la suma sea exactamente 2*k, equivalente a X̂ = Ŷ + Ẑ + Ŵ
            diff = 2 * self.k - sum(c_del_grupo)
            c_del_grupo[0] += diff  # ajustar el primero
            c_del_grupo[0] = max(0, min(c_del_grupo[0], self.k))
            
            # Codificar en bits
            for c_val in c_del_grupo:
                bits = [int(b) for b in f"{c_val:0{self.bits_per_c}b}"]
                cert.extend(bits)
        
        return cert
        
    def _decode_binary_to_c(self, bit_chunk):
        val_int = 0
        for bit in bit_chunk:
            val_int = (val_int << 1) | bit
        return val_int % len(self.valid_c_values)

    def _evaluate_test(self, group_indices, cert):
        c_evaluar = []
        particiones_evaluar = []
        ids_absolutos = []
        
        for g_idx in group_indices:
            grupo = self.grupos[g_idx]
            c_del_grupo = []
            
            for i in range(4):
                logical_idx = g_idx * 4 + i
                ids_absolutos.append(logical_idx)
                
                chunk_start = logical_idx * self.bits_per_c
                chunk = cert[chunk_start : chunk_start + self.bits_per_c]
                c_val = self._decode_binary_to_c(chunk)
                
                c_evaluar.append(c_val)
                c_del_grupo.append(c_val)
                particiones_evaluar.append(grupo['partitions'][i])
                
            # TEST ALGEBRAICO (X̂ = Ŷ + Ẑ + Ŵ, es decir X̂ - Ŷ - Ẑ - Ŵ = 0)
            diferencias = [self.k - 2 * c for c in c_del_grupo]
            if diferencias[0] - diferencias[1] - diferencias[2] - diferencias[3] != 0:
                return 0
                
        total_particiones_leidas = self.t2 * 4
        
        # =========================================================
        # FASE DE RECOLECCIÓN GLOBAL (Uniendo todos los grupos)
        # =========================================================
        global_subsets = []
        global_reqs = []
        
        for idx in range(self.t2):
            for i in range(4):
                # (Asumiendo que usas la versión de nodos directos 'c')
                c_normal = c_evaluar[idx*4 + i]
                c_prime = self.k - c_normal
                P, P_prime = particiones_evaluar[idx*4 + i]
                
                # Descarte físico inicial rápido 
                if c_normal > len(P) or c_prime > len(P_prime) or c_prime < 0 or c_normal < 0:
                    return 0
                global_subsets.extend([P, P_prime])
                global_reqs.extend([c_normal, c_prime])
                
        # =========================================================
        # TEST 1 GLOBAL: Conexiones Recíprocas + Edmonds
        # =========================================================

        total_cajas = len(global_subsets) 
        matchings_por_caja = []
        free_nodes_por_caja = []
        
        for j in range(total_cajas):
            if global_reqs[j] > 0:
                key = frozenset(global_subsets[j])
                matching_j, free_j = self._matching_cache[key]
                matchings_por_caja.append(matching_j)
                free_nodes_por_caja.append(free_j)
            else:
                matchings_por_caja.append(set())
                free_nodes_por_caja.append([])
        # =========================================================
        # 4. TEST TOPOLÓGICO Y CONTEOS EFECTIVOS
        # =========================================================
        for i in range(total_cajas):
            req_i = global_reqs[i]
            if req_i <= 0: continue 
            
            S_i_nodes = list(global_subsets[i])
            valid_nodes = set()
            
            # PASO A: Cribado cruzado (Nodos que cumplen las exigencias)
            for node_u in S_i_nodes:
                is_valid = True
                for j in range(total_cajas):
                    req_j = global_reqs[j]
                    if req_j > 0:
                        vecinos_en_j = self.neighbors_cache[node_u].intersection(global_subsets[j])
                        
                        vecinos_efectivos = 0
                        
                        # 1. Los nodos libres suman 1 si son vecinos
                        for free_node in free_nodes_por_caja[j]:
                            if free_node in vecinos_en_j:
                                vecinos_efectivos += 1
                                
                        # 2. Las parejas congeladas suman MÁXIMO 1 si node_u conoce al menos a uno
                        for v1, v2 in matchings_por_caja[j]:
                            if v1 in vecinos_en_j or v2 in vecinos_en_j:
                                vecinos_efectivos += 1
                                
                        if vecinos_efectivos < req_j:
                            is_valid = False
                            break
                            
                if is_valid:
                    valid_nodes.add(node_u)
            
            # PASO B: Conteo Efectivo Final en la caja i
            effective_valid_count = 0
            
            # 1. Nodos libres que sobrevivieron
            for free_node in free_nodes_por_caja[i]:
                if free_node in valid_nodes:
                    effective_valid_count += 1
                    
            # 2. Parejas congeladas donde AL MENOS UNO sobrevivió
            for v1, v2 in matchings_por_caja[i]:
                if v1 in valid_nodes or v2 in valid_nodes:
                    effective_valid_count += 1
                    
            if effective_valid_count < req_i:
                return 0 
                
        return 1

    def calc_fitness(self, cert):
        if self.total_tests == 0: return 0.0
        passed_tests = sum(self._evaluate_test(test, cert) for test in self.tests_a_evaluar)
        return passed_tests / self.total_tests

    def evaluate_full_web(self, cert):
        """
        Evalúa el certificado completo contra TODOS los grupos de la telaraña a la vez.
        Devuelve 1 si el certificado es globalmente perfecto, o 0 si falla en cualquier punto.
        """
        if self.num_grupos == 0:
            return 0.0
            
        # Creamos una lista con todos los índices posibles [0, 1, 2, ..., num_grupos - 1]
        todos_los_grupos = list(range(self.num_grupos))
        
        # Llamamos a tu función original intacta
        resultado = self._evaluate_test(todos_los_grupos, cert)
        
        return float(resultado)
    
class test_t2grupos_bits_parciales:
    def __init__(self, graph_instance, web, t=4, max_tests=50, W=3):
        """
        t2: Número de GRUPOS que el verificador seleccionará.
        W: Tamaño de la "ventana" de bits de magnitud a leer.
        """
        self.G = graph_instance
        self.t2 = max(1, t // 4)
        self.W = W
        self.grupos = web.group_definitions if hasattr(web, 'group_definitions') else web.groups
        self.num_grupos = len(self.grupos)
        self.k = self.G.target_clique_size
        
        # --- MATEMÁTICA DE LAS DIFERENCIAS ---
        self.M = math.ceil(math.log2(self.k + 1))
        self.W = min(self.W, self.M)
        self.num_bits = self.num_grupos * 4 * self.W 

        # --- CACHÉ ---
        self.neighbors_cache = {u: set(self.G.graph.neighbors(u)) | {u} for u in range(self.G.n)}

        # --- GENERACIÓN DE TESTS ---
        total_posibles = math.comb(self.num_grupos, self.t2)
        if total_posibles <= max_tests:
            self.tests_a_evaluar = list(itertools.combinations(range(self.num_grupos), self.t2))
        else:
            tests_set = set()
            while len(tests_set) < max_tests:
                tests_set.add(tuple(sorted(random.sample(range(self.num_grupos), self.t2))))
            self.tests_a_evaluar = list(tests_set)
            
        self.total_tests = len(self.tests_a_evaluar)
        
        # Pre-calcular matchings para todas las particiones posibles
        self._matching_cache = {}
        for grupo in self.grupos:
            for P, P_prime in grupo['partitions']:
                for subset in [P, P_prime]:
                    key = frozenset(subset)
                    if key not in self._matching_cache:
                        S = list(subset)
                        G_comp = nx.Graph()
                        G_comp.add_nodes_from(S)
                        for idx1 in range(len(S)):
                            for idx2 in range(idx1 + 1, len(S)):
                                u, v = S[idx1], S[idx2]
                                if v not in self.neighbors_cache[u]:
                                    G_comp.add_edge(u, v)
                        matching = nx.max_weight_matching(G_comp, maxcardinality=True)
                        matched = set(n for pair in matching for n in pair)
                        free = [n for n in S if n not in matched]
                        self._matching_cache[key] = (matching, free)
                        
        self.cert_greedy = self._construir_certificado_greedy()

    def _construir_certificado_greedy(self):
        cert = []
        for grupo in self.grupos:
            for P, P_prime in grupo['partitions']:
                # Estimar δ: diferencia entre nodos bien conectados en P vs P'
                grado_P = sum(self.G.graph.degree(u) for u in P) / len(P)
                grado_Pp = sum(self.G.graph.degree(u) for u in P_prime) / len(P_prime)
                total = grado_P + grado_Pp
                c = self.k // 2 if total == 0 else round(self.k * grado_P / total)
                cert.extend(self._codificar_cantidad(c))
        return cert
        
    def _codificar_cantidad(self, c):
        c = max(0, min(int(c), self.k))
        full_bits = [int(b) for b in f"{c:0{self.M}b}"]
        return full_bits[:self.W]
        
    def _decode_intervalo(self, chunk):
        """De los W bits altos, devuelve [x_min, x_max] completando con 0s y 1s."""
        min_bin = chunk + [0] * (self.M - self.W)
        max_bin = chunk + [1] * (self.M - self.W)
        x_min = int("".join(map(str, min_bin)), 2)
        x_max = int("".join(map(str, max_bin)), 2)
        x_min = min(x_min, self.k)
        x_max = min(x_max, self.k)
        return x_min, x_max

    def _evaluate_test(self, group_indices, cert):

        global_subsets = []
        global_reqs = []
        
        # 1. LECTURA Y TEST CONSISTENCIA
        for g_idx in group_indices:
            grupo = self.grupos[g_idx]

            # Intervalos [x_min, x_max] de cada subconjunto primero del grupo
            intervalos = []   # (x_min, x_max) para X, Y, Z, W
            for i in range(4):
                logical_idx = g_idx * 4 + i
                chunk_start = logical_idx * self.W
                chunk = cert[chunk_start: chunk_start + self.W]
                x_min, x_max = self._decode_intervalo(chunk)
                intervalos.append((x_min, x_max))

            # Diferencias
            dif = []
            for (x_min, x_max) in intervalos:
                d_min = self.k - 2 * x_max
                d_max = self.k - 2 * x_min
                dif.append((d_min, d_max))

            (Xmin, Xmax) = dif[0]
            (Ymin, Ymax) = dif[1]
            (Zmin, Zmax) = dif[2]
            (Wmin, Wmax) = dif[3]
            
            # TEST DE CONSISTENCIA: 0 debe estar en el rango de X̂ - Ŷ - Ẑ - Ŵ
            # máximo de (X̂ - Ŷ - Ẑ - Ŵ) = Xmax - (Ymin + Zmin + Wmin)
            # mínimo de (X̂ - Ŷ - Ẑ - Ŵ) = Xmin - (Ymax + Zmax + Wmax)
            sup = Xmax - (Ymin + Zmin + Wmin)
            inf = Xmin - (Ymax + Zmax + Wmax)
            if not (inf <= 0 <= sup):
                return 0
                
            # Requisitos para el test de conexión: el MÍNIMO del intervalo
            for i in range(4):
                x_min, x_max = intervalos[i]
                c_req = x_min                  # mínimo en el primer subconjunto
                c_prime_req = self.k - x_max   # mínimo en el segundo (x' >= k - x_max)
                P, P_prime = grupo['partitions'][i]
                if c_req > len(P) or c_prime_req > len(P_prime):
                    return 0
                global_subsets.extend([P, P_prime])
                global_reqs.extend([max(0, c_req), max(0, c_prime_req)])
                
        # 3. TEST CONEXIÓN
        total_cajas = len(global_subsets) 
        matchings_por_caja = []
        free_nodes_por_caja = []
        
        for j in range(total_cajas):
            if global_reqs[j] > 0:
                key = frozenset(global_subsets[j])
                matching_j, free_j = self._matching_cache[key]
                matchings_por_caja.append(matching_j)
                free_nodes_por_caja.append(free_j)
            else:
                matchings_por_caja.append(set())
                free_nodes_por_caja.append([])
                
        for i in range(total_cajas):
            req_i = global_reqs[i]
            if req_i <= 0: continue 
            
            S_i_nodes = list(global_subsets[i])
            valid_nodes = set()
            
            # PASO A: Cribado cruzado (Nodos que cumplen las exigencias)
            for node_u in S_i_nodes:
                is_valid = True
                for j in range(total_cajas):
                    req_j = global_reqs[j]
                    if req_j > 0:
                        vecinos_en_j = self.neighbors_cache[node_u].intersection(global_subsets[j])
                        
                        vecinos_efectivos = 0
                        
                        # 1. Los nodos libres suman 1 si son vecinos
                        for free_node in free_nodes_por_caja[j]:
                            if free_node in vecinos_en_j:
                                vecinos_efectivos += 1
                                
                        # 2. Las parejas congeladas suman MÁXIMO 1 si node_u conoce al menos a uno
                        for v1, v2 in matchings_por_caja[j]:
                            if v1 in vecinos_en_j or v2 in vecinos_en_j:
                                vecinos_efectivos += 1
                                
                        if vecinos_efectivos < req_j:
                            is_valid = False
                            break
                            
                if is_valid:
                    valid_nodes.add(node_u)
            
            effective_valid_count = 0
            
            for free_node in free_nodes_por_caja[i]:
                if free_node in valid_nodes:
                    effective_valid_count += 1
                    
            for v1, v2 in matchings_por_caja[i]:
                if v1 in valid_nodes or v2 in valid_nodes:
                    effective_valid_count += 1
                    
            if effective_valid_count < req_i:
                return 0 
                
        return 1

    def calc_fitness(self, cert):
        if self.total_tests == 0: return 0.0
        passed_tests = sum(self._evaluate_test(test, cert) for test in self.tests_a_evaluar)
        return passed_tests / self.total_tests
        
    def evaluate_full_web(self, cert):
        """
        Evalúa el certificado completo contra TODOS los grupos de la telaraña a la vez.
        Devuelve 1 si el certificado es globalmente perfecto, o 0 si falla en cualquier punto.
        """
        if self.num_grupos == 0:
            return 0.0
            
        # Creamos una lista con todos los índices posibles [0, 1, 2, ..., num_grupos - 1]
        todos_los_grupos = list(range(self.num_grupos))
        
        # Llamamos a tu función original intacta
        resultado = self._evaluate_test(todos_los_grupos, cert)
        
        return float(resultado)
    
class test3_cantexacta:
    def __init__(self, graph_instance, web, t=4, max_tests=20, k_test=4):
        """
        t2: Número de GRUPOS que el verificador seleccionará en cada test.
        """
        self.G = graph_instance
        self.t2 = t // 4
        self.grupos = web.group_definitions if hasattr(web, 'group_definitions') else web.groups
        self.num_grupos = len(self.grupos)
        self.k = self.G.target_clique_size
        self.k_test = k_test
        
        # Rango de nodos válidos (c)
        self.valid_c_values = list(range(self.k + 1))
        self.bits_per_c = math.ceil(math.log2(len(self.valid_c_values)))
        self.num_bits = self.num_grupos * 4 * self.bits_per_c 
        
        # --- CACHÉS ---
        self.neighbors_cache = {u: set(self.G.graph.neighbors(u)) | {u} for u in range(self.G.n)}
        
        # Mapeamos a qué partición matemática real pertenece cada grupo lógico de bits
        self.partition_ids = []
        unique_partitions = {}
        for g in self.grupos:
            for p, p_prime in g['partitions']:
                key = (frozenset(p), frozenset(p_prime))
                if key not in unique_partitions:
                    unique_partitions[key] = len(unique_partitions)
                self.partition_ids.append(unique_partitions[key])

        # Pre-calcular matchings para todas las particiones posibles
        self._matching_cache = {}
        for grupo in self.grupos:
            for P, P_prime in grupo['partitions']:
                for subset in [P, P_prime]:
                    key = frozenset(subset)
                    if key not in self._matching_cache:
                        S = list(subset)
                        G_comp = nx.Graph()
                        G_comp.add_nodes_from(S)
                        for idx1 in range(len(S)):
                            for idx2 in range(idx1 + 1, len(S)):
                                u, v = S[idx1], S[idx2]
                                if v not in self.neighbors_cache[u]:
                                    G_comp.add_edge(u, v)
                        matching = nx.max_weight_matching(G_comp, maxcardinality=True)
                        matched = set(n for pair in matching for n in pair)
                        free = [n for n in S if n not in matched]
                        self._matching_cache[key] = (matching, free)

        # --- GENERACIÓN DE TESTS (Por Grupos) ---
        total_posibles = math.comb(self.num_grupos, self.t2)
        if total_posibles <= max_tests:
            self.tests_a_evaluar = list(itertools.combinations(range(self.num_grupos), self.t2))
        else:
            tests_set = set()
            while len(tests_set) < max_tests:
                tests_set.add(tuple(sorted(random.sample(range(self.num_grupos), self.t2))))
            self.tests_a_evaluar = list(tests_set)
            
        self.total_tests = len(self.tests_a_evaluar)
        self.cert_greedy = self._construir_certificado_greedy()
        
    def _construir_certificado_greedy(self):
        cert = []
        for grupo in self.grupos:
            c_del_grupo = []
            
            for P, P_prime in grupo['partitions']:
                # Estimar c proporcionalmente al grado medio
                grado_P = sum(self.G.graph.degree(u) for u in P) / len(P)
                grado_Pp = sum(self.G.graph.degree(u) for u in P_prime) / len(P_prime)
                
                total_grado = grado_P + grado_Pp
                if total_grado == 0:
                    c = self.k // 2
                else:
                    c = round(self.k * grado_P / total_grado)
                
                c = max(0, min(c, self.k))
                c_del_grupo.append(c)
            
            # Ajustar para que la suma sea exactamente 2*k, equivalente a X̂ = Ŷ + Ẑ + Ŵ
            diff = 2 * self.k - sum(c_del_grupo)
            c_del_grupo[0] += diff  # ajustar el primero
            c_del_grupo[0] = max(0, min(c_del_grupo[0], self.k))
            
            # Codificar en bits
            for c_val in c_del_grupo:
                bits = [int(b) for b in f"{c_val:0{self.bits_per_c}b}"]
                cert.extend(bits)
        
        return cert
        
    def _decode_binary_to_c(self, bit_chunk):
        val_int = 0
        for bit in bit_chunk:
            val_int = (val_int << 1) | bit
        return val_int % len(self.valid_c_values)

    def _evaluate_test(self, group_indices, cert):
        c_evaluar = []
        particiones_evaluar = []
        ids_absolutos = []
        
        for g_idx in group_indices:
            grupo = self.grupos[g_idx]
            c_del_grupo = []
            
            for i in range(4):
                logical_idx = g_idx * 4 + i
                ids_absolutos.append(logical_idx)
                
                chunk_start = logical_idx * self.bits_per_c
                chunk = cert[chunk_start : chunk_start + self.bits_per_c]
                c_val = self._decode_binary_to_c(chunk)
                
                c_evaluar.append(c_val)
                c_del_grupo.append(c_val)
                particiones_evaluar.append(grupo['partitions'][i])
                
            # TEST CONSISTENCIA (X̂ = Ŷ + Ẑ + Ŵ, es decir X̂ - Ŷ - Ẑ - Ŵ = 0)
            diferencias = [self.k - 2 * c for c in c_del_grupo]
            if diferencias[0] - diferencias[1] - diferencias[2] - diferencias[3] != 0:
                return 0
                
        total_particiones_leidas = self.t2 * 4
        
        # =========================================================
        # FASE DE RECOLECCIÓN GLOBAL (Uniendo todos los grupos)
        # =========================================================
        global_subsets = []
        global_reqs = []
        
        for idx in range(self.t2):
            for i in range(4):
                # (Asumiendo que usas la versión de nodos directos 'c')
                c_normal = c_evaluar[idx*4 + i]
                c_prime = self.k - c_normal
                P, P_prime = particiones_evaluar[idx*4 + i]
                
                # Descarte físico inicial rápido 
                if c_normal > len(P) or c_prime > len(P_prime) or c_prime < 0 or c_normal < 0:
                    return 0
                global_subsets.extend([P, P_prime])
                global_reqs.extend([c_normal, c_prime])

        total_cajas = len(global_subsets) 
        matchings_por_caja = []
        free_nodes_por_caja = []
        
        for j in range(total_cajas):
            if global_reqs[j] > 0:
                key = frozenset(global_subsets[j])
                matching_j, free_j = self._matching_cache[key]
                matchings_por_caja.append(matching_j)
                free_nodes_por_caja.append(free_j)
            else:
                matchings_por_caja.append(set())
                free_nodes_por_caja.append([])
        # =========================================================
        # TEST CONEXIÓN Y CONTEOS EFECTIVOS
        # =========================================================
        for i in range(total_cajas):
            req_i = global_reqs[i]
            if req_i <= 0: continue 
            
            S_i_nodes = list(global_subsets[i])
            valid_nodes = set()
            
            for node_u in S_i_nodes:
                is_valid = True
                for j in range(total_cajas):
                    req_j = global_reqs[j]
                    if req_j > 0:
                        vecinos_en_j = self.neighbors_cache[node_u].intersection(global_subsets[j])
                        
                        vecinos_efectivos = 0
                        
                        # 1. Los nodos libres suman 1 si son vecinos
                        for free_node in free_nodes_por_caja[j]:
                            if free_node in vecinos_en_j:
                                vecinos_efectivos += 1
                                
                        # 2. Las parejas congeladas suman MÁXIMO 1 si node_u conoce al menos a uno
                        for v1, v2 in matchings_por_caja[j]:
                            if v1 in vecinos_en_j or v2 in vecinos_en_j:
                                vecinos_efectivos += 1
                                
                        if vecinos_efectivos < req_j:
                            is_valid = False
                            break
                            
                if is_valid:
                    valid_nodes.add(node_u)
            
            # Conteo Efectivo Final en la caja i
            effective_valid_count = 0
            
            for free_node in free_nodes_por_caja[i]:
                if free_node in valid_nodes:
                    effective_valid_count += 1
                    
            for v1, v2 in matchings_por_caja[i]:
                if v1 in valid_nodes or v2 in valid_nodes:
                    effective_valid_count += 1
                    
            if effective_valid_count < req_i:
                return 0 

        # =========================================================
        # TEST 3: Muestreo probabilístico en el subconjunto vencedor
        # =========================================================
        idx_vencedor = max(range(len(global_reqs)), key=lambda j: global_reqs[j])
        subconjunto_vencedor = list(global_subsets[idx_vencedor])

        # Elegimos 8 nodos uniformemente (sin reemplazo si hay >= 8, con
        # reemplazo o todos si hay menos).
        num_muestra = 8
        if len(subconjunto_vencedor) >= num_muestra:
            muestra = random.sample(subconjunto_vencedor, num_muestra)
        else:
            muestra = subconjunto_vencedor  # si hay menos de 8, tomamos todos

        # Comprobamos si entre los nodos de la muestra existe un k-clique.
        if not self._existe_k_clique(muestra, self.k_test):
            return 0

        return 1

    def _existe_k_clique(self, nodos, k_test):
        if len(nodos) < k_test:
            return False
        H = self.G.graph.subgraph(nodos)
        clique, _ = nx.max_weight_clique(H, weight=None)
        return len(clique) >= k_test
        
    def calc_fitness(self, cert):
        if self.total_tests == 0: return 0.0
        passed_tests = sum(self._evaluate_test(test, cert) for test in self.tests_a_evaluar)
        return passed_tests / self.total_tests

    def evaluate_full_web(self, cert):
        """
        Evalúa el certificado completo contra TODOS los grupos de la telaraña a la vez.
        Devuelve 1 si el certificado es globalmente perfecto, o 0 si falla en cualquier punto.
        """
        if self.num_grupos == 0:
            return 0.0
            
        # Creamos una lista con todos los índices posibles [0, 1, 2, ..., num_grupos - 1]
        todos_los_grupos = list(range(self.num_grupos))
        
        # Llamamos a tu función original intacta
        resultado = self._evaluate_test(todos_los_grupos, cert)
        
        return float(resultado)
    
class test3_margencant:
    def __init__(self, graph_instance, web, t=4, max_tests=50, W=3, k_test=4):
        """
        t2: Número de GRUPOS que el verificador seleccionará.
        W: Tamaño de la "ventana" de bits de magnitud a leer.
        """
        self.G = graph_instance
        self.t2 = max(1, t // 4)
        self.W = W
        self.grupos = web.group_definitions if hasattr(web, 'group_definitions') else web.groups
        self.num_grupos = len(self.grupos)
        self.k = self.G.target_clique_size
        self.k_test=k_test
        
        # --- MATEMÁTICA DE LAS DIFERENCIAS ---
        self.M = math.ceil(math.log2(self.k + 1))
        self.W = min(self.W, self.M)
        self.num_bits = self.num_grupos * 4 * self.W 

        # --- CACHÉ ---
        self.neighbors_cache = {u: set(self.G.graph.neighbors(u)) | {u} for u in range(self.G.n)}

        # --- GENERACIÓN DE TESTS ---
        total_posibles = math.comb(self.num_grupos, self.t2)
        if total_posibles <= max_tests:
            self.tests_a_evaluar = list(itertools.combinations(range(self.num_grupos), self.t2))
        else:
            tests_set = set()
            while len(tests_set) < max_tests:
                tests_set.add(tuple(sorted(random.sample(range(self.num_grupos), self.t2))))
            self.tests_a_evaluar = list(tests_set)
            
        self.total_tests = len(self.tests_a_evaluar)
        
        # Pre-calcular matchings para todas las particiones posibles
        self._matching_cache = {}
        for grupo in self.grupos:
            for P, P_prime in grupo['partitions']:
                for subset in [P, P_prime]:
                    key = frozenset(subset)
                    if key not in self._matching_cache:
                        S = list(subset)
                        G_comp = nx.Graph()
                        G_comp.add_nodes_from(S)
                        for idx1 in range(len(S)):
                            for idx2 in range(idx1 + 1, len(S)):
                                u, v = S[idx1], S[idx2]
                                if v not in self.neighbors_cache[u]:
                                    G_comp.add_edge(u, v)
                        matching = nx.max_weight_matching(G_comp, maxcardinality=True)
                        matched = set(n for pair in matching for n in pair)
                        free = [n for n in S if n not in matched]
                        self._matching_cache[key] = (matching, free)
                        
        self.cert_greedy = self._construir_certificado_greedy()

    def _construir_certificado_greedy(self):
        cert = []
        for grupo in self.grupos:
            for P, P_prime in grupo['partitions']:
                # Estimar δ: diferencia entre nodos bien conectados en P vs P'
                grado_P = sum(self.G.graph.degree(u) for u in P) / len(P)
                grado_Pp = sum(self.G.graph.degree(u) for u in P_prime) / len(P_prime)
                total = grado_P + grado_Pp
                c = self.k // 2 if total == 0 else round(self.k * grado_P / total)
                cert.extend(self._codificar_cantidad(c))
        return cert
        
    def _codificar_cantidad(self, c):
        c = max(0, min(int(c), self.k))
        full_bits = [int(b) for b in f"{c:0{self.M}b}"]
        return full_bits[:self.W]
        
    def _decode_intervalo(self, chunk):
        """De los W bits altos, devuelve [x_min, x_max] completando con 0s y 1s."""
        min_bin = chunk + [0] * (self.M - self.W)
        max_bin = chunk + [1] * (self.M - self.W)
        x_min = int("".join(map(str, min_bin)), 2)
        x_max = int("".join(map(str, max_bin)), 2)
        x_min = min(x_min, self.k)
        x_max = min(x_max, self.k)
        return x_min, x_max

    def _evaluate_test(self, group_indices, cert):

        global_subsets = []
        global_reqs = []
        
        # 1. LECTURA Y TEST CONSISTENCIA
        for g_idx in group_indices:
            grupo = self.grupos[g_idx]

            # Intervalos [x_min, x_max] de cada subconjunto primero del grupo
            intervalos = []   # (x_min, x_max) para X, Y, Z, W
            for i in range(4):
                logical_idx = g_idx * 4 + i
                chunk_start = logical_idx * self.W
                chunk = cert[chunk_start: chunk_start + self.W]
                x_min, x_max = self._decode_intervalo(chunk)
                intervalos.append((x_min, x_max))

            # Diferencias
            dif = []
            for (x_min, x_max) in intervalos:
                d_min = self.k - 2 * x_max
                d_max = self.k - 2 * x_min
                dif.append((d_min, d_max))

            (Xmin, Xmax) = dif[0]
            (Ymin, Ymax) = dif[1]
            (Zmin, Zmax) = dif[2]
            (Wmin, Wmax) = dif[3]
            
            # TEST DE CONSISTENCIA: 0 debe estar en el rango de X̂ - Ŷ - Ẑ - Ŵ
            # máximo de (X̂ - Ŷ - Ẑ - Ŵ) = Xmax - (Ymin + Zmin + Wmin)
            # mínimo de (X̂ - Ŷ - Ẑ - Ŵ) = Xmin - (Ymax + Zmax + Wmax)
            sup = Xmax - (Ymin + Zmin + Wmin)
            inf = Xmin - (Ymax + Zmax + Wmax)
            if not (inf <= 0 <= sup):
                return 0
                
            # Requisitos para el test de conexión: el MÍNIMO del intervalo
            for i in range(4):
                x_min, x_max = intervalos[i]
                c_req = x_min                  # mínimo en el primer subconjunto
                c_prime_req = self.k - x_max   # mínimo en el segundo (x' >= k - x_max)
                P, P_prime = grupo['partitions'][i]
                if c_req > len(P) or c_prime_req > len(P_prime):
                    return 0
                global_subsets.extend([P, P_prime])
                global_reqs.extend([max(0, c_req), max(0, c_prime_req)])
                
        # 3. TEST CONEXIÓN
        total_cajas = len(global_subsets) 
        matchings_por_caja = []
        free_nodes_por_caja = []
        
        for j in range(total_cajas):
            if global_reqs[j] > 0:
                key = frozenset(global_subsets[j])
                matching_j, free_j = self._matching_cache[key]
                matchings_por_caja.append(matching_j)
                free_nodes_por_caja.append(free_j)
            else:
                matchings_por_caja.append(set())
                free_nodes_por_caja.append([])
                
        for i in range(total_cajas):
            req_i = global_reqs[i]
            if req_i <= 0: continue 
            
            S_i_nodes = list(global_subsets[i])
            valid_nodes = set()
            
            # PASO A: Cribado cruzado (Nodos que cumplen las exigencias)
            for node_u in S_i_nodes:
                is_valid = True
                for j in range(total_cajas):
                    req_j = global_reqs[j]
                    if req_j > 0:
                        vecinos_en_j = self.neighbors_cache[node_u].intersection(global_subsets[j])
                        
                        vecinos_efectivos = 0
                        
                        # 1. Los nodos libres suman 1 si son vecinos
                        for free_node in free_nodes_por_caja[j]:
                            if free_node in vecinos_en_j:
                                vecinos_efectivos += 1
                                
                        # 2. Las parejas congeladas suman MÁXIMO 1 si node_u conoce al menos a uno
                        for v1, v2 in matchings_por_caja[j]:
                            if v1 in vecinos_en_j or v2 in vecinos_en_j:
                                vecinos_efectivos += 1
                                
                        if vecinos_efectivos < req_j:
                            is_valid = False
                            break
                            
                if is_valid:
                    valid_nodes.add(node_u)
            
            effective_valid_count = 0
            
            for free_node in free_nodes_por_caja[i]:
                if free_node in valid_nodes:
                    effective_valid_count += 1
                    
            for v1, v2 in matchings_por_caja[i]:
                if v1 in valid_nodes or v2 in valid_nodes:
                    effective_valid_count += 1
                    
            if effective_valid_count < req_i:
                return 0 

        # =========================================================
        # TEST 3: Muestreo probabilístico en el subconjunto vencedor
        # =========================================================
        idx_vencedor = max(range(len(global_reqs)), key=lambda j: global_reqs[j])
        subconjunto_vencedor = list(global_subsets[idx_vencedor])

        # Elegimos 8 nodos uniformemente (sin reemplazo si hay >= 8, con
        # reemplazo o todos si hay menos).
        num_muestra = 8
        if len(subconjunto_vencedor) >= num_muestra:
            muestra = random.sample(subconjunto_vencedor, num_muestra)
        else:
            muestra = subconjunto_vencedor  # si hay menos de 8, tomamos todos

        # Comprobamos si entre los nodos de la muestra existe un k-clique.
        if not self._existe_k_clique(muestra, self.k_test):
            return 0

        return 1

    def _existe_k_clique(self, nodos, k_test):
        if len(nodos) < k_test:
            return False
        H = self.G.graph.subgraph(nodos)
        clique, _ = nx.max_weight_clique(H, weight=None)
        return len(clique) >= k_test
        
    def calc_fitness(self, cert):
        if self.total_tests == 0: return 0.0
        passed_tests = sum(self._evaluate_test(test, cert) for test in self.tests_a_evaluar)
        return passed_tests / self.total_tests
        
    def evaluate_full_web(self, cert):
        """
        Evalúa el certificado completo contra TODOS los grupos de la telaraña a la vez.
        Devuelve 1 si el certificado es globalmente perfecto, o 0 si falla en cualquier punto.
        """
        if self.num_grupos == 0:
            return 0.0
            
        # Creamos una lista con todos los índices posibles [0, 1, 2, ..., num_grupos - 1]
        todos_los_grupos = list(range(self.num_grupos))
        
        # Llamamos a tu función original intacta
        resultado = self._evaluate_test(todos_los_grupos, cert)
        
        return float(resultado)
    
class test_4_reciprocidad_logn:
    def __init__(self, graph_instance, num_universos=20, consultas=4, epsilon=0.05):
        self.G = graph_instance
        self.n = self.G.target_clique_size
        self.num_universos = num_universos
        self.consultas = min(consultas, num_universos)
        self.epsilon = epsilon

        self.log_n = max(1, math.ceil(math.log2(self.n)))
        self.cte_asintotica = 0.63  # 1 - 1/e

        # Certificado: cantidad EXACTA de éxitos por universo (de momento)
        self.bits_per_prop = math.ceil(math.log2(self.n + 1))
        self.num_bits = self.num_universos * self.bits_per_prop

        self.neighbors_cache = {u: set(self.G.graph.neighbors(u)) for u in range(self.G.n)}

        # Caché de la tasa real de éxito ya calculada por universo
        # clave: idx del universo -> tasa_real_exito (float)
        self._cache_tasa_real = {}
        self._cache_cliques = {}

        # Generación de universos
        self.universos = []
        for _ in range(self.num_universos):
            A = []
            for _ in range(self.n):
                B = []
                for _ in range(self.n):
                    X = tuple(random.sample(range(self.G.n), self.log_n))
                    B.append(X)
                A.append(B)
            self.universos.append(A)

        # Tests a evaluar
        total_posibles = math.comb(self.num_universos, self.consultas)
        max_tests = 20
        if total_posibles <= max_tests:
            self.tests_a_evaluar = list(itertools.combinations(range(self.num_universos), self.consultas))
        else:
            tests_set = set()
            while len(tests_set) < max_tests:
                tests_set.add(tuple(sorted(random.sample(range(self.num_universos), self.consultas))))
            self.tests_a_evaluar = list(tests_set)

        self.total_tests = len(self.tests_a_evaluar)

    def _es_clique(self, nodos):
        for i in range(len(nodos)):
            for j in range(i + 1, len(nodos)):
                if nodos[j] not in self.neighbors_cache[nodos[i]]:
                    return False
        return True

    def _estan_conectados(self, X, Y):
        for u in X:
            for v in Y:
                if u != v and v not in self.neighbors_cache[u]:
                    return False
        return True

    def _decode_prop(self, chunk):
        val = int("".join(map(str, chunk)), 2)
        return min(val, self.n) / self.n

    def _precalcular_cliques(self, idx_universo):
        """
        Para el universo dado, devuelve una lista 'cliques_por_B' donde
        cliques_por_B[i] es la lista de semillas X de B_i que son cliques.
        Cachea el resultado.
        """
        if idx_universo in self._cache_cliques:
            return self._cache_cliques[idx_universo]

        A_target = self.universos[idx_universo]
        cliques_por_B = []
        for B_i in A_target:
            cliques_B = [X for X in B_i if self._es_clique(X)]
            cliques_por_B.append(cliques_B)

        self._cache_cliques[idx_universo] = cliques_por_B
        return cliques_por_B

    def _calcular_tasa_real(self, idx_universo):
        if idx_universo in self._cache_tasa_real:
            return self._cache_tasa_real[idx_universo]

        cliques_por_B = self._precalcular_cliques(idx_universo)
        umbral_reciprocidad = self.cte_asintotica - self.epsilon
        n_B = len(cliques_por_B)

        exitos_A = 0
        for idx_B, cliques_X in enumerate(cliques_por_B):
            if not cliques_X:
                continue  # B_i no tiene ninguna semilla clique

            encontrado_X_valido = False
            for X in cliques_X:
                conexiones_exitosas = 0
                for idx_Bj, cliques_Y in enumerate(cliques_por_B):
                    if idx_B == idx_Bj:
                        continue
                    # Solo recorremos los Y que YA sabemos que son cliques
                    for Y in cliques_Y:
                        if self._estan_conectados(X, Y):
                            conexiones_exitosas += 1
                            break
                prop_reciprocidad = conexiones_exitosas / (self.n - 1)
                if prop_reciprocidad > umbral_reciprocidad:
                    encontrado_X_valido = True
                    break
            if encontrado_X_valido:
                exitos_A += 1

        tasa_real = exitos_A / self.n
        self._cache_tasa_real[idx_universo] = tasa_real
        return tasa_real

    def _evaluate_test(self, indices_consultados, cert):
        # 1. LEER CERTIFICADO O(1)
        proporciones_leidas = []
        for idx in indices_consultados:
            start = idx * self.bits_per_prop
            chunk = cert[start : start + self.bits_per_prop]
            p_claim = self._decode_prop(chunk)
            proporciones_leidas.append((p_claim, idx))

        # 2. SELECCIÓN DEL MEJOR UNIVERSO (mayor proporción declarada)
        mejor_p_claim, mejor_idx = max(proporciones_leidas, key=lambda x: x[0])

        # 3. UMBRAL EXIGIDO (según el texto):
        #    si la proporción declarada > 0.63  -> exigir p_claim - epsilon
        #    si la proporción declarada <= 0.63 -> exigir 0.63 - epsilon
        if mejor_p_claim > self.cte_asintotica:
            umbral_exigido = mejor_p_claim - self.epsilon
        else:
            umbral_exigido = self.cte_asintotica - self.epsilon

        # 4. TEST PESADO (con caché)
        tasa_real_exito = self._calcular_tasa_real(mejor_idx)

        # 5. VEREDICTO
        return 1 if tasa_real_exito >= umbral_exigido else 0

    def calc_fitness(self, cert):
        if self.total_tests == 0:
            return 0.0
        passed_tests = sum(self._evaluate_test(test, cert) for test in self.tests_a_evaluar)
        return passed_tests / self.total_tests
    
class test_4_reciprocidad_logn_margen:
    def __init__(self, graph_instance, num_universos=20, consultas=4, epsilon=0.05, W=3):
        self.G = graph_instance
        self.n = self.G.n
        self.num_universos = num_universos
        self.consultas = min(consultas, num_universos)
        self.epsilon = epsilon

        self.log_n = max(1, math.ceil(math.log2(self.n)))
        self.cte_asintotica = 0.63  # 1 - 1/e

        # Bits totales para representar la cantidad exacta en [0, n]
        self.M = math.ceil(math.log2(self.n + 1))
        # Guardamos solo los W bits MÁS significativos (PCP estricto, bloque constante)
        self.W = min(W, self.M)
        self.bits_per_prop = self.W
        self.num_bits = self.num_universos * self.bits_per_prop

        self.neighbors_cache = {u: set(self.G.graph.neighbors(u)) for u in range(self.G.n)}

        self._cache_tasa_real = {}
        self._cache_cliques = {}

        # Generación de universos
        self.universos = []
        for _ in range(self.num_universos):
            A = []
            for _ in range(self.n):
                B = []
                for _ in range(self.n):
                    X = tuple(random.sample(range(self.G.n), self.log_n))
                    B.append(X)
                A.append(B)
            self.universos.append(A)

        # Tests a evaluar
        total_posibles = math.comb(self.num_universos, self.consultas)
        max_tests = 50
        if total_posibles <= max_tests:
            self.tests_a_evaluar = list(itertools.combinations(range(self.num_universos), self.consultas))
        else:
            tests_set = set()
            while len(tests_set) < max_tests:
                tests_set.add(tuple(sorted(random.sample(range(self.num_universos), self.consultas))))
            self.tests_a_evaluar = list(tests_set)

        self.total_tests = len(self.tests_a_evaluar)

    def _es_clique(self, nodos):
        for i in range(len(nodos)):
            for j in range(i + 1, len(nodos)):
                if nodos[j] not in self.neighbors_cache[nodos[i]]:
                    return False
        return True

    def _estan_conectados(self, X, Y):
        for u in X:
            for v in Y:
                if u != v and v not in self.neighbors_cache[u]:
                    return False
        return True

    def _decode_prop(self, chunk):
        """
        Interpreta los W bits como los MÁS significativos de un número de M bits,
        completando con ceros los M-W bits menos significativos (cota inferior).
        Devuelve la proporción c/n.
        """
        bits_completos = list(chunk) + [0] * (self.M - self.W)
        val = int("".join(map(str, bits_completos)), 2)
        return min(val, self.n) / self.n

    def _precalcular_cliques(self, idx_universo):
        if idx_universo in self._cache_cliques:
            return self._cache_cliques[idx_universo]
        A_target = self.universos[idx_universo]
        cliques_por_B = []
        for B_i in A_target:
            cliques_B = [X for X in B_i if self._es_clique(X)]
            cliques_por_B.append(cliques_B)
        self._cache_cliques[idx_universo] = cliques_por_B
        return cliques_por_B

    def _calcular_tasa_real(self, idx_universo):
        if idx_universo in self._cache_tasa_real:
            return self._cache_tasa_real[idx_universo]

        cliques_por_B = self._precalcular_cliques(idx_universo)
        umbral_reciprocidad = self.cte_asintotica - self.epsilon

        exitos_A = 0
        for idx_B, cliques_X in enumerate(cliques_por_B):
            if not cliques_X:
                continue
            encontrado_X_valido = False
            for X in cliques_X:
                conexiones_exitosas = 0
                for idx_Bj, cliques_Y in enumerate(cliques_por_B):
                    if idx_B == idx_Bj:
                        continue
                    for Y in cliques_Y:
                        if self._estan_conectados(X, Y):
                            conexiones_exitosas += 1
                            break
                prop_reciprocidad = conexiones_exitosas / (self.n - 1)
                if prop_reciprocidad > umbral_reciprocidad:
                    encontrado_X_valido = True
                    break
            if encontrado_X_valido:
                exitos_A += 1

        tasa_real = exitos_A / self.n
        self._cache_tasa_real[idx_universo] = tasa_real
        return tasa_real

    def _evaluate_test(self, indices_consultados, cert):
        proporciones_leidas = []
        for idx in indices_consultados:
            start = idx * self.bits_per_prop
            chunk = cert[start : start + self.bits_per_prop]
            p_claim = self._decode_prop(chunk)
            proporciones_leidas.append((p_claim, idx))

        mejor_p_claim, mejor_idx = max(proporciones_leidas, key=lambda x: x[0])

        if mejor_p_claim > self.cte_asintotica:
            umbral_exigido = mejor_p_claim - self.epsilon
        else:
            umbral_exigido = self.cte_asintotica - self.epsilon

        tasa_real_exito = self._calcular_tasa_real(mejor_idx)
        return 1 if tasa_real_exito >= umbral_exigido else 0

    def calc_fitness(self, cert):
        if self.total_tests == 0:
            return 0.0
        passed_tests = sum(self._evaluate_test(test, cert) for test in self.tests_a_evaluar)
        return passed_tests / self.total_tests