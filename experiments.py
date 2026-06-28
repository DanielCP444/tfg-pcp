import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import math
from constructions import GraphInstance, SpiderWeb, PureGeneticAlgorithm

#Exhaustivo
def run_experiment(test, n_graphs=500, n_nodes=8, steps=4, t_bits=4, clique=False, filtro_edmonds = True):
    """
    Laboratorio universal de PCP.
    Recibe la clase del test (TesterClass) y evalúa el Algoritmo Genético contra ella.
    """
    print(f"Iniciando Experimento PCP (N={n_nodes}, {steps} pasos de telaraña)...")
    print(f"Test inyectado: {test.__name__} (leyendo {t_bits} bits)")
    
    total_por_aristas = defaultdict(int)
    cazados_por_inverso = defaultdict(int)
    cazados_por_pcp = defaultdict(int)
    fp_por_aristas = defaultdict(int) 
    peor_por_aristas = defaultdict(float) 
    
    # Extra: Vamos a guardar el fitness medio que logra el mentiroso para cada densidad
    fitness_acumulado = defaultdict(float)

    resultados = []
    
    for i in range(n_graphs):
        if i % 10 == 0: 
            print(f"Procesando grafo {i}/{n_graphs}...", end='\r')
        
        # 1. Generar Grafo Negativo (Sin Clique)
        if clique: 
            p_random = random.uniform(0.3, 0.85)
        else: p_random = random.uniform(0.65, 0.95)
        G_obj = GraphInstance(n_nodes, p=p_random, has_clique=clique)
        m_edges = G_obj.graph.number_of_edges()
        total_por_aristas[m_edges] += 1
        if m_edges not in peor_por_aristas:
            peor_por_aristas[m_edges]=0.0
        #Filtro Edmonds
        if filtro_edmonds and G_obj.max_matching_size > (n_nodes - G_obj.target_clique_size):
            mejor_fitness = 0.0
            cazados_por_inverso[m_edges] += 1
            continue
        
        mejor_fitness = 1.0
        # 3. Construir la Telaraña
        web = SpiderWeb(n_nodes)
        web.construir(n_divergencias=steps, converger_cada=4)
        
        # 4. Instanciar el Test Inyectado 
        evaluador = test(G_obj, web, t=t_bits)
        
        num_bits=evaluador.num_bits
        fitness = 0.0
        for cert in itertools.product([0, 1], repeat=num_bits):
            f=evaluador.calc_fitness(cert)
            if f > fitness:
                fitness = f
            if fitness == 1.0:
                break
                
        if fitness < mejor_fitness:
            mejor_fitness = fitness

        fitness_acumulado[m_edges] += mejor_fitness
        
        if mejor_fitness == 1.0:
            fp_por_aristas[m_edges] += 1
        else:
            cazados_por_pcp[m_edges] += 1   
            
        if mejor_fitness*100 > peor_por_aristas[m_edges]:
            peor_por_aristas[m_edges]=mejor_fitness*100

    print("\nExperimento finalizado. Generando gráficas...")
    plot_injected_results(total_por_aristas, cazados_por_inverso, cazados_por_pcp, fp_por_aristas, fitness_acumulado, peor_por_aristas, test.__name__, clique)    

#Algoritmo genético
def run_genetic_experiment_2(test, n_graphs=500, n_nodes=8, steps=4, t_bits=4, clique=False, filtro_edmonds = True):
    """
    Laboratorio universal de PCP.
    Recibe la clase del test (TesterClass) y evalúa el Algoritmo Genético contra ella.
    """
    print(f"Iniciando Experimento PCP (N={n_nodes}, {steps} pasos de telaraña)...")
    print(f"Test inyectado: {test.__name__} (leyendo {t_bits} bits)")
    
    total_por_aristas = defaultdict(int)
    cazados_por_inverso = defaultdict(int)
    cazados_por_pcp = defaultdict(int)
    fp_por_aristas = defaultdict(int) 
    peor_por_aristas = defaultdict(float)
    # Extra: Vamos a guardar el fitness medio que logra el mentiroso para cada densidad
    fitness_acumulado = defaultdict(float)

    resultados = []
    
    for i in range(n_graphs):
        if i % 10 == 0: 
            print(f"Procesando grafo {i}/{n_graphs}...", end='\r')
        
        # 1. Generar Grafo
        if clique: 
            p_random = random.uniform(0.4, 0.85)
        else: p_random = random.uniform(0.75, 0.95)
        G_obj = GraphInstance(n_nodes, p=p_random, has_clique=clique)
        # Métricas del grafo
        m_edges = G_obj.graph.number_of_edges()
        grados = [d for _, d in G_obj.graph.degree()]
        varianza_grado = np.var(grados)
        clique_max, _ = nx.algorithms.clique.max_weight_clique(G_obj.graph, weight=None)
        tam_clique_max = len(clique_max)
        matching_comp = G_obj.max_matching_size
        total_por_aristas[m_edges] += 1
        mejor_fitness = 1.0
        if m_edges not in peor_por_aristas:
            peor_por_aristas[m_edges]=0.0
        # 2. Filtro Edmonds (El guardián global)
        if filtro_edmonds and G_obj.max_matching_size > (n_nodes - G_obj.target_clique_size):
            mejor_fitness = 0.0
            cazados_por_inverso[m_edges] += 1
            continue
        
        # 3. Construir la Telaraña Desenrollada
        web = SpiderWeb(n_nodes)
        web.construir(n_divergencias=steps, converger_cada=4)
        
        # 4. Instanciar el Test Inyectado (Inyección de Dependencias)
        # Le pasamos el grafo, la web y el parámetro t
        evaluador = test(G_obj, web, t=t_bits)
        
        seed = getattr(evaluador, 'cert_greedy', None)
        
        # 5. Desatar al Algoritmo Genético
        genetico = PureGeneticAlgorithm(
            num_bits=evaluador.num_bits, 
            fitness_evaluator=evaluador.calc_fitness,
            pop_size=100, 
            generations=50,
            seed_individual=seed
        )
        fitness, _ = genetico.run()
        if fitness < mejor_fitness:
            mejor_fitness = fitness

        resultados.append({
            'aristas': m_edges,
            'varianza_grado': varianza_grado,
            'clique_max': tam_clique_max,
            'matching_comp': matching_comp,
            'fitness': mejor_fitness
        })

        fitness_acumulado[m_edges] += mejor_fitness
        
        if mejor_fitness == 1.0:
            fp_por_aristas[m_edges] += 1
        else:
            cazados_por_pcp[m_edges] += 1
        if mejor_fitness*100 > peor_por_aristas[m_edges]:
            peor_por_aristas[m_edges]=mejor_fitness*100  

    print("\nExperimento finalizado. Generando gráficas...")
    plot_injected_results(total_por_aristas, cazados_por_inverso, cazados_por_pcp, fp_por_aristas, fitness_acumulado, peor_por_aristas, test.__name__, clique)
    df = pd.DataFrame(resultados)
    
    # Correlaciones con el fitness
    print("\nCorrelaciones con el fitness:")
    print(df.corr()['fitness'].sort_values(ascending=False))
    
    # Scatter plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for ax, col in zip(axes, ['aristas', 'varianza_grado', 'clique_max', 'matching_comp']):
        ax.scatter(df[col], df['fitness'], alpha=0.5, s=10)
        ax.set_xlabel(col)
        ax.set_ylabel('fitness')
        ax.set_title(f'{col} vs fitness')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()    """
    Laboratorio universal de PCP.
    Recibe la clase del test (TesterClass) y evalúa el Algoritmo Genético contra ella.
    """
    print(f"Iniciando Experimento PCP (N={n_nodes}, {steps} pasos de telaraña)...")
    print(f"Test inyectado: {test.__name__} (leyendo {t_bits} bits)")
    
    total_por_aristas = defaultdict(int)
    cazados_por_inverso = defaultdict(int)
    cazados_por_pcp = defaultdict(int)
    fp_por_aristas = defaultdict(int) 
    peor_por_aristas = defaultdict(float)
    # Extra: Vamos a guardar el fitness medio que logra el mentiroso para cada densidad
    fitness_acumulado = defaultdict(float)

    resultados = []
    
    for i in range(n_graphs):
        if i % 10 == 0: 
            print(f"Procesando grafo {i}/{n_graphs}...", end='\r')
        
        # 1. Generar Grafo
        if clique: 
            p_random = random.uniform(0.4, 0.85)
        else: p_random = random.uniform(0.75, 0.95)
        G_obj = GraphInstance(n_nodes, p=p_random, has_clique=clique)
        # Métricas del grafo
        m_edges = G_obj.graph.number_of_edges()
        grados = [d for _, d in G_obj.graph.degree()]
        varianza_grado = np.var(grados)
        clique_max, _ = nx.algorithms.clique.max_weight_clique(G_obj.graph, weight=None)
        tam_clique_max = len(clique_max)
        matching_comp = G_obj.max_matching_size
        total_por_aristas[m_edges] += 1
        mejor_fitness = 1.0
        if m_edges not in peor_por_aristas:
            peor_por_aristas[m_edges]=0.0
        # 2. Filtro Edmonds (El guardián global)
        if filtro_edmonds and G_obj.max_matching_size > (n_nodes - G_obj.target_clique_size):
            mejor_fitness = 0.0
            cazados_por_inverso[m_edges] += 1
            continue
        
        for j in range(1):
            # 3. Construir la Telaraña Desenrollada
            web = SpiderWeb2(n_nodes)
            web.expand_web(steps)
            
            # 4. Instanciar el Test Inyectado (Inyección de Dependencias)
            # Le pasamos el grafo, la web y el parámetro t
            evaluador = test(G_obj, web, t=t_bits)
            
            seed = getattr(evaluador, 'cert_greedy', None)
            
            # 5. Desatar al Algoritmo Genético
            genetico = PureGeneticAlgorithm(
                num_bits=evaluador.num_bits, 
                fitness_evaluator=evaluador.calc_fitness,
                pop_size=100, 
                generations=50,
                seed_individual=seed
            )
            fitness, _ = genetico.run()
            if fitness < mejor_fitness:
                mejor_fitness = fitness

        resultados.append({
            'aristas': m_edges,
            'varianza_grado': varianza_grado,
            'clique_max': tam_clique_max,
            'matching_comp': matching_comp,
            'fitness': mejor_fitness
        })

        fitness_acumulado[m_edges] += mejor_fitness
        
        if mejor_fitness == 1.0:
            fp_por_aristas[m_edges] += 1
        else:
            cazados_por_pcp[m_edges] += 1
        if mejor_fitness*100 > peor_por_aristas[m_edges]:
            peor_por_aristas[m_edges]=mejor_fitness*100  

    print("\nExperimento finalizado. Generando gráficas...")
    plot_injected_results(total_por_aristas, cazados_por_inverso, cazados_por_pcp, fp_por_aristas, fitness_acumulado, peor_por_aristas, test.__name__, clique)
    df = pd.DataFrame(resultados)
    
    # Correlaciones con el fitness
    print("\nCorrelaciones con el fitness:")
    print(df.corr()['fitness'].sort_values(ascending=False))
    
    # Scatter plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for ax, col in zip(axes, ['aristas', 'varianza_grado', 'clique_max', 'matching_comp']):
        ax.scatter(df[col], df['fitness'], alpha=0.5, s=10)
        ax.set_xlabel(col)
        ax.set_ylabel('fitness')
        ax.set_title(f'{col} vs fitness')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

#Exhaustivo
def run_test_experiment(test, n_graphs=50, steps=4, t_bits=4, clique=False, filtro_edmonds = True):
    """
    Laboratorio universal de PCP.
    Recibe la clase del test (TesterClass) y evalúa el Algoritmo Genético contra ella.
    """
    print(f"Iniciando Experimento PCP (N=[8..100], {steps} pasos de telaraña)...")
    print(f"Test inyectado: {test.__name__} (leyendo {t_bits} bits)")
    
    total_por_nodos = defaultdict(int)
    cazados_por_inverso = defaultdict(int)
    cazados_por_pcp = defaultdict(int)
    fp_por_nodos = defaultdict(int)
    peor_por_nodos = defaultdict(float)
    # Extra: Vamos a guardar el fitness medio que logra el mentiroso para cada densidad
    fitness_acumulado = defaultdict(float)

    resultados = []
    n_nodes = [8,16,24,36,50,66,82,100]

    for n in n_nodes:
        fitness_acumulado[n] = 0.0
        total_por_nodos[n]=1
        peor_por_nodos[n]=0.0
        for i in range(n_graphs):
            if i % 10 == 0: 
                print(f"Procesando grafo de {n} nodos {i}/{n_graphs}...", end='\r')
            p_random = random.uniform(0.65, 0.95)
            G_obj = GraphInstance(n, p=p_random, has_clique=clique)

            # 2. Filtro Edmonds (El guardián global)
            if filtro_edmonds and G_obj.max_matching_size > (n - G_obj.target_clique_size):
                mejor_fitness = 0.0
                cazados_por_inverso[n] += 1
                continue
            total_por_nodos[n] +=1
            mejor_fitness = 1.0
            web = SpiderWeb(n_nodes)
            web.construir(n_divergencias=steps, converger_cada=4)
            
            # 4. Instanciar el Test Inyectado (Inyección de Dependencias)
            evaluador = test(G_obj, web, t=t_bits)
            
            num_bits=evaluador.num_bits
            fitness = 0.0
            for cert in itertools.product([0, 1], repeat=num_bits):
                f=evaluador.calc_fitness(cert)
                if f > fitness:
                    fitness = f
                if fitness == 1.0:
                    break
                    
            if fitness < mejor_fitness:
                mejor_fitness = fitness
    
            fitness_acumulado[n] += mejor_fitness
            
            if mejor_fitness == 1.0:
                fp_por_nodos[n] += 1
            else:
                cazados_por_pcp[n] += 1
                
            if mejor_fitness*100 > peor_por_nodos[n]:
                peor_por_nodos[n]=mejor_fitness*100
            
    print("\nExperimento finalizado. Generando gráficas...")
    plot_injected_results(total_por_nodos, cazados_por_inverso, cazados_por_pcp, fp_por_nodos, fitness_acumulado, peor_por_nodos, test.__name__, clique)

#Algoritmo genético
def run_test_genetic_experiment_2(test, n_graphs=50, t_bits=4, clique=False, filtro_edmonds = True):
    """
    Laboratorio universal de PCP.
    Recibe la clase del test (TesterClass) y evalúa el Algoritmo Genético contra ella.
    """
    print(f"Iniciando Experimento PCP (N=[8..100], pasos de telaraña crecientes)...")
    print(f"Test inyectado: {test.__name__} (leyendo {t_bits} bits)")
    
    total_por_nodos = defaultdict(int)
    cazados_por_inverso = defaultdict(int)
    cazados_por_pcp = defaultdict(int)
    fp_por_nodos = defaultdict(int)
    sin_edmonds_por_nodos = defaultdict(int)
    peor_por_nodos = defaultdict(float)
    # Extra: Vamos a guardar el fitness medio que logra el mentiroso para cada densidad
    fitness_acumulado = defaultdict(float)

    resultados = []
    n_nodes = [8,16,24,36,50,66,82,100]
    steps = [8,10,12,14,16,18,20,24]

    for k,n in enumerate(n_nodes):
        fitness_acumulado[n] = 0.0
        total_por_nodos[n]=n_graphs
        peor_por_nodos[n]=0.0
        sin_edmonds_por_nodos[n]=0
        for i in range(n_graphs):
            if i % 10 == 0: 
                print(f"Procesando grafo de {n} nodos {i}/{n_graphs}...", end='\r')
            p_random = random.uniform(0.75, 0.95)
            G_obj = GraphInstance(n, p=p_random, has_clique=clique)

            # 2. Filtro Edmonds (El guardián global)
            if filtro_edmonds and G_obj.max_matching_size > (n - G_obj.target_clique_size):
                mejor_fitness = 0.0
                cazados_por_inverso[n] += 1
                continue
            sin_edmonds_por_nodos[n] +=1
            mejor_fitness = 1.0
            for j in range(4):
                web = SpiderWeb(n)
                web.construir(n_divergencias=steps[k])
                
                # 4. Instanciar el Test Inyectado (Inyección de Dependencias)
                evaluador = test(G_obj, web, t=t_bits)
                
                seed = getattr(evaluador, 'cert_greedy', None)
                
                # 5. Desatar al Algoritmo Genético
                genetico = PureGeneticAlgorithm(
                    num_bits=evaluador.num_bits, 
                    fitness_evaluator=evaluador.calc_fitness,
                    pop_size=100, 
                    generations=50,
                    seed_individual=seed
                )
                fitness, _ = genetico.run()
                        
                if fitness < mejor_fitness:
                    mejor_fitness = fitness
    
            fitness_acumulado[n] += mejor_fitness
            
            if mejor_fitness == 1.0:
                fp_por_nodos[n] += 1
            else:
                cazados_por_pcp[n] += 1
                
            if mejor_fitness*100 > peor_por_nodos[n]:
                peor_por_nodos[n]=mejor_fitness*100
            
    print("\nExperimento finalizado. Generando gráficas...")
    plot_injected_results_test(total_por_nodos, cazados_por_inverso, cazados_por_pcp, fp_por_nodos, fitness_acumulado, peor_por_nodos, test.__name__, clique)

def plot_injected_results(total_dict, inv_dict, pcp_dict, fp_dict, fitness_dict, peor_fitness, test_name, clique=False):
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
    ax1.set_xlabel("Número de Aristas (m)")
    ax1.set_ylabel("Porcentaje de Casos (%)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Gráfica 2: Fitness medio
    color_fitness = 'green' if clique else 'purple'
    ax2.plot(aristas_sorted, avg_fitness, marker='o', color=color_fitness, linestyle='-', linewidth=2)
    ax2.fill_between(aristas_sorted, avg_fitness, color=color_fitness, alpha=0.1)
    ax2.set_title(titulo_fitness)
    ax2.set_xlabel("Número de Aristas (m)")
    ax2.set_ylabel("Fitness Medio (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_injected_results_test(total_dict, inv_dict, pcp_dict, fp_dict, fitness_dict, peor_fitness, test_name, clique=False):
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

def generar_certificado_honesto_exacto(clique_real, web, graph_instance):
    """
    Genera el certificado honesto para el test de cantidades exactas.
    Para cada 2-partición (P, P') codifica el número de nodos del clique
    que contiene el primer subconjunto P.
    """
    clique = set(clique_real)
    k = graph_instance.target_clique_size
    bits_per_c = math.ceil(math.log2(k + 1))

    grupos = web.group_definitions if hasattr(web, 'group_definitions') else web.groups

    cert = []
    for grupo in grupos:
        for P, P_prime in grupo['partitions']:
            c = len(clique & set(P))          # nodos del clique en el primer subconjunto
            c = max(0, min(c, k))             # seguridad: dentro de [0, k]
            bits = [int(b) for b in f"{c:0{bits_per_c}b}"]
            cert.extend(bits)

    return cert

def generar_certificado_honesto_margen(clique_real, web, graph_instance, W=3):
    """
    Genera el certificado honesto para el test de margen de cantidades.
    Para cada 2-partición (P, P') codifica los W bits MÁS significativos
    de la cantidad de nodos del clique que contiene el primer subconjunto P.
    """
    clique = set(clique_real)
    k = graph_instance.target_clique_size
    M = math.ceil(math.log2(k + 1))   # bits totales para representar c en [0, k]
    W = min(W, M)                      # por si W es mayor que M

    grupos = web.group_definitions if hasattr(web, 'group_definitions') else web.groups

    cert = []
    for grupo in grupos:
        for P, P_prime in grupo['partitions']:
            c = len(clique & set(P))           # nodos del clique en el primer subconjunto
            c = max(0, min(c, k))              # dentro de [0, k]
            full_bits = [int(b) for b in f"{c:0{M}b}"]   # c en M bits
            cert.extend(full_bits[:W])         # los W más significativos
    return cert

def test_completitud_exacto(test,n_graphs=100,n_nodes=16, steps=4):
    print("--- INICIANDO TEST DE COMPLETITUD (MODELO DE NODOS 'c') ---")
    nota_acumulada = 0
    for i in range(n_graphs):
        # 1. Generar grafo POSITIVO (con clique garantizado)
        G_obj = GraphInstance(n_nodes, p=0.7, has_clique=True)
        clique_real_set = G_obj.clique_nodes 
        #print(f"Clique real de tamaño {G_obj.target_clique_size} inyectado.")
        #print(f"Nodos del clique: {clique_real_set}")
        
        # 2. Construir la Telaraña
        web = SpiderWeb2(n_nodes)
        web.expand_web(steps)
        grupos_web = web.group_definitions if hasattr(web, 'group_definitions') else getattr(web, 'groups', [])
        #print(f"Telaraña generada con {len(grupos_web)} grupos.")
        
        # 3. El Demostrador Honesto redacta el certificado
        cert_honesto = generar_certificado_honesto_exacto(clique_real_set, web, G_obj)
        #print(f"Certificado honesto generado (Longitud: {len(cert_honesto)} bits).")
        
        # 4. Instanciamos el Verificador Maestro (el que usa 'c')
        # Ajusta el nombre de la clase al que estés usando en tu código
        verificador = test(G_obj, web, t=4, max_tests=50)
        
        # 5. Pasamos el examen
        nota_final = verificador.calc_fitness(cert_honesto)
        nota_acumulada+=nota_final
    
    print("\nRESULTADO FINAL:")
    print(f"Fitness del Demostrador Honesto: {nota_acumulada/n_graphs * 100}%")
    
    # ATENCIÓN A LA TEORÍA AQUÍ:
    if nota_final == 1.0:
        print("¡ÉXITO! Completitud perfecta (1.0). El verificador acepta la verdad.")
    else:
        print("ALERTA: El verificador ha rechazado una verdad. Revisa la topología.")

def test_completitud_margen(test,n_graphs=100):
    print("--- INICIANDO TEST DE COMPLETITUD (MODELO DE NODOS 'c') ---")
    nota_acumulada = 0
    fn = 0
    n_nodes=[16,24,36,50]
    steps_n=[6,7,8,10]
    for k,n in enumerate(n_nodes):
        steps=steps_n[k]        
        for i in range(n_graphs):
            # 1. Generar grafo POSITIVO (con clique garantizado)
            p=random.uniform(0.5,0.95)
            G_obj = GraphInstance(n, p=p, has_clique=True)
            clique_real_set = G_obj.clique_nodes 
            #print(f"Clique real de tamaño {G_obj.target_clique_size} inyectado.")
            #print(f"Nodos del clique: {clique_real_set}")
            
            # 2. Construir la Telaraña
            web = SpiderWeb(n)
            web.construir(steps)
            grupos_web = web.group_definitions if hasattr(web, 'group_definitions') else getattr(web, 'groups', [])
            #print(f"Telaraña generada con {len(grupos_web)} grupos.")
            
            # 3. El Demostrador Honesto redacta el certificado
            cert_honesto = generar_certificado_honesto_margen(clique_real_set, web, G_obj)
            #print(f"Certificado honesto generado (Longitud: {len(cert_honesto)} bits).")
            
            # 4. Instanciamos el Verificador Maestro (el que usa 'c')
            # Ajusta el nombre de la clase al que estés usando en tu código
            verificador = test(G_obj, web, t=4, max_tests=50)
            
            # 5. Pasamos el examen
            nota_final = verificador.calc_fitness(cert_honesto)
            nota_acumulada+=nota_final
            if nota_final < 1:
                fn+=1
    
    print("\nRESULTADO FINAL:")
    print(f"Fitness del Demostrador Honesto: {nota_acumulada/n_graphs/4 * 100}%")
    print(f"Falsos negativos: {fn/n_graphs/4 * 100}%")
    
def generar_certificado_honesto_test4(evaluador, clique_real):
    """
    Genera el certificado honesto para el test 4.
    Para cada universo A_i, cuenta cuántos B_j contienen al menos un
    conjunto C_{jl} totalmente dentro del clique C, y escribe los W bits
    MÁS significativos de esa cantidad.
    """
    clique = set(clique_real)
    M = evaluador.M          # bits totales para representar la cantidad en [0, n]
    W = evaluador.W          # bits significativos que se guardan
    n = evaluador.n

    cert = []
    for A_i in evaluador.universos:
        # Contar cuántos B_j tienen al menos un C_{jl} ⊂ C
        exitos = 0
        for B_j in A_i:
            if any(set(C_jl) <= clique for C_jl in B_j):
                exitos += 1
        exitos = max(0, min(exitos, n))           # dentro de [0, n]
        full_bits = [int(b) for b in f"{exitos:0{M}b}"]   # cantidad en M bits
        cert.extend(full_bits[:W])                # los W más significativos
    return cert

def test_completitud_test4(n_graphs=100, num_universos=20, consultas=4, epsilon=0.05, W=3):
    """
    Comprueba la completitud del test 4: en grafos con clique, mide qué
    fracción de veces el certificado honesto alcanza fitness 1.
    """
    print("--- COMPLETITUD TEST 4 (reciprocidad O(log n)) ---")
    print(f"Universos={num_universos}, consultas={consultas}, "
          f"epsilon={epsilon}, W={W}, trials={n_graphs}\n")

    fitness_total = 0.0
    completos = 0   # cuántos trials dan fitness exactamente 1.0
    n_nodes=[16,24,36,50]

    for n in n_nodes:
        for t in range(n_graphs):
            # 1. Grafo positivo (con clique)
            p_random=random.uniform(0.7,0.95)
            G_obj = GraphInstance(n, p=p_random, has_clique=True)
            clique_real = G_obj.clique_nodes
    
            # 2. Evaluador del test 4
            evaluador = test_4_reciprocidad_logn_margen(
                G_obj,
                num_universos=num_universos,
                consultas=consultas,
                epsilon=epsilon,
                W=W
            )
    
            # 3. Certificado honesto
            cert = generar_certificado_honesto_test4(evaluador, clique_real)
    
            # 4. Fitness
            fit = evaluador.calc_fitness(cert)
            fitness_total += fit
            if fit == 1.0:
                completos += 1
    
            #print(f"  Trial {t+1:>2}: fitness = {fit:.3f}")

    completitud_media = fitness_total / n_graphs/4
    frac_perfectos = completos / n_graphs/4

    print(f"\nCompletitud media (fitness honesto promedio): {completitud_media:.3f}")
    print(f"Fracción de trials con fitness perfecto (1.0): {frac_perfectos:.3f}")
    return completitud_media, frac_perfectos

#Transformación de grafos de la sección 4.7
def transform_G_to_G_prime(G, epsilon=0.05):
    """
    Transforma el grafo G en G' usando la técnica de reciprocidad asintótica (50%).
    Recibe directamente un objeto nx.Graph.
    """
    n = G.number_of_nodes()
    
    # Calculamos el n' para clavar el 50% de probabilidad
    n_prime = max(1, int(n * 0.693)) 
    log_n = max(1, math.ceil(math.log2(n)))
    
    neighbors_cache = {u: set(G.neighbors(u)) for u in range(n)}
    
    def es_clique(nodos):
        for i in range(len(nodos)):
            for j in range(i+1, len(nodos)):
                if nodos[j] not in neighbors_cache[nodos[i]]: return False
        return True
        
    def estan_conectados(X, Y):
        for u in X:
            for v in Y:
                if u != v and v not in neighbors_cache[u]: return False
        return True
    
    # Construir A (n bolsas B_i, cada una con n' semillas X)
    A = []
    for _ in range(n):
        B = [tuple(random.sample(range(n), log_n)) for _ in range(n_prime)]
        A.append(B)
        
    # Pre-filtrar semillas que ya son cliques internos
    X_validos_por_B = {}
    for i, B in enumerate(A):
        X_validos_por_B[i] = [X for X in B if es_clique(X)]
        
    # Identificar el conjunto BB (Filtro de Reciprocidad > 0.5 - epsilon)
    BB_indices = set()
    
    for i in range(n):
        if not X_validos_por_B[i]: continue
            
        encontrado_fuerte = False
        for X in X_validos_por_B[i]:
            conexiones_exitosas = 0
            for j in range(n):
                if i == j: continue
                for Y in X_validos_por_B[j]:
                    if estan_conectados(X, Y):
                        conexiones_exitosas += 1
                        break
                        
            if (conexiones_exitosas / (n - 1)) > (0.5 - epsilon):
                encontrado_fuerte = True
                break
                
        if encontrado_fuerte:
            BB_indices.add(i)
            
    # Construir G'
    G_prime = nx.Graph()
    G_prime.add_nodes_from(range(n))
    
    BB_list = list(BB_indices)
    for idx1 in range(len(BB_list)):
        for idx2 in range(idx1 + 1, len(BB_list)):
            i = BB_list[idx1]
            j = BB_list[idx2]
            
            arista_creada = False
            for X in X_validos_por_B[i]:
                for Y in X_validos_por_B[j]:
                    if estan_conectados(X, Y):
                        G_prime.add_edge(i, j)
                        arista_creada = True
                        break
                if arista_creada: break
                    
    return G_prime

def run_density_trap_experiment(n_nodes=16, num_trials=3, clique=False):
    print("==================================================")
    print(f"🔍 EXPERIMENTO: LA TRAMPA DE LA DENSIDAD (N={n_nodes})")
    print("Comparando Aristas de G vs G' según la probabilidad")
    print("==================================================\n")
    
    probabilidades = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    max_aristas = (n_nodes * (n_nodes - 1)) // 2
    
    resultados_G = []
    resultados_G_prime = []
    
    print(f"{'Prob. (p)':<10} | {'Aristas en G':<15} | {'Aristas en G':<15}'")
    print("-" * 45)
    
    for p in probabilidades:
        media_G = 0
        media_G_prime = 0
        
        for _ in range(num_trials):
            # Generamos un grafo aleatorio puro (Erdos-Renyi)
            G_obj = GraphInstance(n_nodes, p=p, has_clique=clique)
            G = nx.erdos_renyi_graph(n_nodes, p)
            G_prime = transform_G_to_G_prime(G)
            
            media_G += G.number_of_edges()
            media_G_prime += G_prime.number_of_edges()
            
        media_G /= num_trials
        media_G_prime /= num_trials
        
        resultados_G.append(media_G)
        resultados_G_prime.append(media_G_prime)
        
        print(f"{p:<10} | {media_G:>6.1f} / {max_aristas:<6} | {media_G_prime:>6.1f} / {max_aristas:<6}")
        
    print("=" * 45)
    plot_density_trap(probabilidades, resultados_G, resultados_G_prime, max_aristas)

def plot_density_trap(probs, edges_G, edges_G_prime, max_edges):
    plt.figure(figsize=(10, 6))
    
    plt.plot(probs, edges_G, marker='o', color='dodgerblue', linewidth=2, label="Grafo Original (G)")
    plt.plot(probs, edges_G_prime, marker='s', color='crimson', linewidth=2, label="Grafo Transformado (G')")
    
    # Línea de máximo posible de aristas
    plt.axhline(y=max_edges, color='black', linestyle='--', alpha=0.5, label="Máximo Teórico de Aristas")
    
    plt.title("Evolución de la Densidad Topológica: G vs G'")
    plt.xlabel("Probabilidad de Arista en G (p)")
    plt.ylabel("Número Medio de Aristas")
    plt.ylim(-5, max_edges + 5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

