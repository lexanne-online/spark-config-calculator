import math
import streamlit as st
import pandas as pd
from scipy.optimize import fsolve
from streamlit_extras.buy_me_a_coffee import button

def convert_to_megabytes(size_str):
    if size_str[-1] == 'k':
        multiplier = 1024
    elif size_str[-1] == 'm':
        multiplier = 1048576
    elif size_str[-1] == 'g':
        multiplier = 1073741824
    else:
        st.error("Invalid size format")
        exit(-1)

    numeric_value = float(size_str[:-1]) * multiplier / 1024 / 1024
    return numeric_value

def solve_equation(yarn_memory_mb, spark_executor_memory_overhead_percent, spark_offheap_memory, num_executors_per_node):
    initial_guess = 0.0
    p =  spark_executor_memory_overhead_percent
    x =  yarn_memory_mb
    y =  num_executors_per_node
    m =  spark_offheap_memory

    try:        
        spark_onheap_memory = int(fsolve(lambda z: z + max(384, p * z) - (x / y - m), initial_guess)[0])
    except ZeroDivisionError:
        st.error("Number of executors per node cannot be zero")
        exit(-1)
    return spark_onheap_memory


def set_page_header_format():
    st.set_page_config(
    page_title="Spark Configuration Calculator",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
    )

    st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=0, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )

    navbar = """
        <style>
            .navbar {
                background-color: #333;
                padding: 10px;
                color: white;
                text-align: center;
                top-margin: 0px;
                
            }
            .navbar a {
                color: white;
                text-decoration: none;
                padding: 10px;
            }
        </style>
        <div class="navbar">
            <a href="https://www.jeromerajan.com">About</a>
            <a href="https://linkedin.com/in/jeromerajan">LinkedIn</a>
            <a href="https://medium.com/@datasherlock">Medium</a>
            <a href="https://github.com/datasherlock">Github</a>
            <a href="https://buymeacoffee.com/datasherlock">Buy me a coffee</a>
        </div>
    """
    
    st.markdown("<h1 style='text-align: center; '>Spark Configuration Calculator</h1>", unsafe_allow_html=True)
    st.markdown(navbar, unsafe_allow_html=True)

    
    # columns = st.columns(8)

    # with columns[2]:
    #     st.write("""<div style="width:100%;text-align:center;"><a href="https://www.jeromerajan.com" style="float:center"><img src="https://cdn0.iconfinder.com/data/icons/england-13/504/sherlock-holmes-detective-inspector-man-512.png" width="22px"></img></a></div>""", unsafe_allow_html=True)

    
    # with columns[3]:
    #     st.write("""<div style="width:100%;text-align:center;"><a href="https://linkedin.com/in/jeromerajan" style="float:center"><img src="https://cdn2.iconfinder.com/data/icons/social-media-applications/64/social_media_applications_14-linkedin-512.png" width="22px"></img></a></div>""", unsafe_allow_html=True)
        
    # with columns[4]:
    #     st.write("""<div style="width:100%;text-align:center;"><a href="https://medium.com/@datasherlock" style="float:center"><img src="https://cdn2.iconfinder.com/data/icons/social-icons-33/128/Medium-512.png" width="22px"></img></a></div>""", unsafe_allow_html=True)
        
    # with columns[5]:
    #     st.write("""<div style="width:100%;text-align:center;"><a href="https://github.com/datasherlock" style="float:center"><img src="https://cdn3.iconfinder.com/data/icons/social-media-2169/24/social_media_social_media_logo_github_2-512.png" width="22px"></img></a></div>""", unsafe_allow_html=True)
    
    st.markdown("""---""")
    #_, feedback, _ = st.columns(3)
    #feedback.markdown("""Share your feedback at [Github Repo Issues](https://github.com/datasherlock/spark-config-calculator/issues)""")

def revised_recommendations(num_workers, capacity_scheduler, cores_per_node, yarn_memory_mb, total_yarn_memory_mb, spark_executor_cores, spark_executor_memory_overhead_percent, spark_memory_fraction, spark_memory_storage_fraction, spark_offheap_memory, spark_submit_deploy_mode, spark_onheap_memory, revised, total_physical_cores):
    with revised:
        if capacity_scheduler == "Default Resource Calculator":
            revised_spark_executor_memory = convert_to_megabytes(st.text_input("Revised spark.executor.memory", value=str(spark_onheap_memory) + 'm'))
            revised_num_executors_per_node_memory = math.floor(yarn_memory_mb/ (spark_offheap_memory  + spark_executor_memory_overhead_percent*revised_spark_executor_memory + revised_spark_executor_memory))
            revised_num_executors = max(0,revised_num_executors_per_node_memory*num_workers) if spark_submit_deploy_mode == "client" else max(0,revised_num_executors_per_node_memory*num_workers) - 1
            

            if revised_num_executors > 0:
                if math.floor(yarn_memory_mb%(spark_offheap_memory  + spark_executor_memory_overhead_percent*revised_spark_executor_memory + revised_spark_executor_memory)) > 0:
                    st.warning(f"{math.floor(yarn_memory_mb%(spark_offheap_memory  + spark_executor_memory_overhead_percent*revised_spark_executor_memory + revised_spark_executor_memory))}  MiB will be unused per node resulting in inefficient utilisation")
    
                df_revised = create_recommendations_matrix(spark_executor_cores, spark_executor_memory_overhead_percent, \
                                                        spark_memory_fraction, spark_memory_storage_fraction, \
                                                            spark_offheap_memory, spark_submit_deploy_mode, \
                                                                revised_spark_executor_memory, revised_num_executors)
                
                job_submission_display_tabs(df_revised)
                
                st.write(df_revised)
                revised_memory_utilised = round(revised_spark_executor_memory * revised_num_executors_per_node_memory* num_workers ,2)
                revised_cores_utilised = spark_executor_cores * revised_num_executors_per_node_memory* num_workers 
                physical_cores = cores_per_node*num_workers
                    
                display_utilisation_scorecard(total_yarn_memory_mb, revised_memory_utilised, revised_cores_utilised, total_physical_cores)
                
            else:
                st.warning("No executors can be allocated with the current configurations. Please tune the parameters")
            
                
        else:
            st.write("Not applicable for DominantResourceCalculator")

def job_submission_display_tabs(df_revised):
    spark_submit, dp_submit = construct_dataproc_submit_command(df_revised)
    spark_submit_tab, dp_submit_tab = st.tabs(["Spark Submit Command", "Dataproc Submit Command"])
    with spark_submit_tab:
        st.code(spark_submit, language="bash")
    with dp_submit_tab:
        st.code(dp_submit, language="bash")

def generate_spark_submit_command(spark_executor_cores, spark_executor_memory_overhead_percent, spark_memory_fraction, spark_memory_storage_fraction, spark_offheap_memory, spark_submit_deploy_mode, spark_executor_memory, num_executors):

    command = f"spark-submit --class {spark_submit_deploy_mode} --master yarn --deploy-mode {spark_submit_deploy_mode} \
        --executor-memory {int(spark_executor_memory)}m --executor-cores {spark_executor_cores} --num-executors {num_executors} \
            --conf spark.memory.fraction={spark_memory_fraction} --conf spark.memory.storageFraction={spark_memory_storage_fraction} \
                --conf spark.memory.offHeap.size={spark_offheap_memory} --conf spark.executor.memoryOverheadFactor={spark_executor_memory_overhead_percent} \
                    --conf spark.dynamicAllocation.initialExecutors={num_executors} --conf spark.dynamicAllocation.enabled=true"
    
    return command




def create_recommendations_matrix(spark_executor_cores, spark_executor_memory_overhead_percent, spark_memory_fraction, spark_memory_storage_fraction, spark_offheap_memory, spark_submit_deploy_mode, spark_executor_memory, num_executors):
    
    return pd.DataFrame({
                        'Recommended Spark Configurations': [
                    f"spark.executor.memory = {int(spark_executor_memory)}m",
                    f"spark.executor.cores= {spark_executor_cores}",
                    f"spark.memory.fraction= {spark_memory_fraction}",
                    f"spark.memory.storageFraction= {spark_memory_storage_fraction}",
                    f"spark.memory.offHeap.size= {spark_offheap_memory}",
                    f"spark.executor.memoryOverheadFactor= {spark_executor_memory_overhead_percent}",
                    f"spark.dynamicAllocation.initialExecutors = {num_executors}",
                    f"spark.dynamicAllocation.enabled=true",
                    f"spark.submit.deployMode={spark_submit_deploy_mode}",
                    f"spark.sql.adaptive.enabled=true",
                    f"spark.serializer=org.apache.spark.serializer.KryoSerializer"
                    ], 'Explanation' : [
                    f"Amount of memory to use per executor process",
                    "The number of cores to use on each executor in Yarn.  \n  In standalone mode, this will equal all the cores in a node",
                    f"Fraction of (heap space - 300MB) used for execution and storage. The lower this is, the more frequently spills and cached data eviction occur. The purpose of this config is to set aside memory for internal metadata, user data structures, and imprecise size estimation in the case of sparse, unusually large records.",
                    f"Amount of storage memory immune to eviction, expressed as a fraction of the size of the region set aside by spark.memory.fraction. The higher this is, the less working memory may be available to execution and tasks may spill to disk more often",
                    f"The absolute amount of memory which can be used for off-heap allocation, in bytes unless otherwise specified. This setting has no impact on heap memory usage, so if your executors' total memory consumption must fit within some hard limit then be sure to shrink your JVM heap size accordingly. ",
                    f"Fraction of executor memory to be allocated as additional non-heap memory per executor process. This is memory that accounts for things like VM overheads, interned strings, other native overheads, etc. This tends to grow with the container size.",
                    f"Initial number of executors to run if dynamic allocation is enabled.",
                    f"Whether to use dynamic resource allocation, which scales the number of executors registered with this application up and down based on the workload.",
                    f"Whether to run in client or cluster mode. Cluster mode will run the AM in a Yarn container while client mode will run the AM in the master node",
                    f"Adaptive Query Execution (AQE) is an optimization technique in Spark SQL that makes use of the runtime statistics to choose the most efficient query execution plan, which is enabled by default since Apache Spark 3.2.0",
                    f"This setting configures the serializer used for not only shuffling data between worker nodes but also when serializing RDDs to disk. This is recommended over the Java serializer"
                    ]
                    })

def memory_breakdown_guidance(total_yarn_memory_mb, memory_breakdown, storage_memory, execution_memory, user_memory, total_memory_utilised, total_cores_utilised, total_physical_cores):
    with memory_breakdown:
        st.write(pd.DataFrame({
                'Memory Type' : [
                    f"""Storage Memory (in MiB) = {storage_memory}""",
                    f"""Execution Memory (in MiB) = {execution_memory}""",
                    f"User memory (in MiB) = {user_memory}",
                    f"Reserved Memory (in MiB) = 300"
                ], 
                'Explanation': [
                    "Used for caching, broadcast variables, etc. Supports spilling to disk",
                    """Used for intermediate data during shuffles, joins, etc. 
                    Disk spill supported. 
                    Can borrow space from Storage memory. May possibly end up with smaller storage space than initial calculations""",
                    "Used to store data structures. Potential cause for OOMs",
                    "Set aside a fixed amount of memory for non-storage, non-execution purposes"
                ]
            }))


        display_utilisation_scorecard(total_yarn_memory_mb, total_memory_utilised, total_cores_utilised, total_physical_cores)


        

def display_utilisation_scorecard(total_yarn_memory_mb, total_memory_utilised, total_cores_utilised, total_physical_cores):
    utilisation_container = st.container(border=True)
    title , memory, cpu = utilisation_container.columns(3)
    title.markdown("<h5 style='text-align: center; vertical-align: middle;'>Utilisation Scorecard</h5>", unsafe_allow_html=True)
    memory.metric("Memory Utilisation", f"{total_memory_utilised}m" , f"{round((total_memory_utilised - total_yarn_memory_mb)/total_yarn_memory_mb*100,2)}%", help="There will always be some under-utilisation since a minimum of 384 MiB overhead is required" )
    cpu.metric("CPU Utilisation", f"{total_cores_utilised} vcores" , f"{round((total_cores_utilised - total_physical_cores)/total_physical_cores*100,2)}%" )

def recommendations(num_workers, cores_per_node, reserve_core, total_yarn_memory_mb, spark_executor_cores, spark_executor_memory_overhead_percent, spark_memory_fraction, spark_memory_storage_fraction, spark_offheap_memory, spark_submit_deploy_mode, num_executors_per_node, spark_onheap_memory, spark_num_executors, results):
    
    with results:
        if spark_num_executors > 0:
            
            df = create_recommendations_matrix(spark_executor_cores, spark_executor_memory_overhead_percent, \
                                            spark_memory_fraction, spark_memory_storage_fraction, \
                                                spark_offheap_memory, spark_submit_deploy_mode, \
                                                    spark_onheap_memory, spark_num_executors)
            
            job_submission_display_tabs(df)            
            st.write(df)
        else:
            st.warning("No executors can be allocated with the current configurations. Please tune the parameters")

        storage_memory = round((spark_onheap_memory - 300) * spark_memory_storage_fraction * spark_memory_fraction,2)
        execution_memory = round((spark_onheap_memory - 300) * spark_memory_fraction * (1 - spark_memory_storage_fraction),2)
        user_memory = round(spark_onheap_memory - (storage_memory + execution_memory) - 300,2)

        total_memory_utilised = round(spark_onheap_memory * num_executors_per_node * num_workers,2)
        total_cores_utilised = spark_executor_cores * num_executors_per_node * num_workers
        total_physical_cores = (cores_per_node - 1)*num_workers if reserve_core == "Yes" else cores_per_node * num_workers
        

        
        display_utilisation_scorecard(total_yarn_memory_mb, total_memory_utilised, total_cores_utilised, total_physical_cores)
    return storage_memory,execution_memory,user_memory,total_memory_utilised,total_cores_utilised,total_physical_cores



def spark_executor_config(executor, num_workers, capacity_scheduler, yarn_cpu_vcores, yarn_memory_mb):
    with executor:
        spark_executor_cores = st.number_input("spark.executor.cores (cores per executor)", 1, yarn_cpu_vcores, \
                                                   help=""" 
                                                   - This is the number of parallel tasks in each executor (Recommended to keep under 5 for Yarn)  
                                                   - This defaults to 1 when using YARN and defaults to all available cores in a node when using standalone
                                                   """, value=min(yarn_cpu_vcores, 5))
        
        if yarn_cpu_vcores%spark_executor_cores != 0:
            if capacity_scheduler == 'Dominant Resource Calculator':
                st.warning(f"Setting `spark.executor.cores` to {spark_executor_cores} will leave {yarn_cpu_vcores%spark_executor_cores} core/s per node unused & result in under-utilisation of CPU") 
        
        spark_executor_memory_overhead_percent = (st.number_input("spark.executor.memoryOverheadFactor %", 0, 100, value = 6, help="""6%-10% ideally. Allocate more as executor size increases. This is done as non-JVM tasks need more non-JVM heap space and such tasks commonly fail with "Memory Overhead Exceeded" errors.""") / 100)
        spark_memory_fraction = st.number_input("spark.memory.fraction", value=0.6, help="The default is 0.6 in Spark 3.x. Recommended to use the default")
        spark_memory_storage_fraction = st.number_input("spark.memory.storageFraction", value=0.5, help="The default is 0.5 in Spark 3.x. Recommended to use the default")
        spark_offheap_memory = st.number_input("spark.memory.offHeap.size (in MB)", value=0, help="The absolute amount of memory which can be used for off-heap allocation") 

        spark_submit_deploy_mode = st.selectbox("spark.submit.deployMode", options = ["client", "cluster"], help= """ \
                                                    - The default mode in Dataproc is `client`
                                                    - `client` mode will schedule the driver and AM on the master node
                                                    - Using the `cluster` mode will mean that the AM will be scheduled in one of the worker nodes
                                                    - One executor will be reserved for the AM in cluster mode. `n-1` executors will be available for actual processing
                                                    - In the `cluster` mode, the Spark driver runs the job in the spark-submit process, and Spark logs are sent to the Dataproc job driver.
                                                    - In the `client` mode, the Spark driver runs the job in a YARN container. Spark driver logs are not available to the Dataproc job driver.
                                                    - This is a nice matrix to understand the different implications of this property on logging behavior - https://cloud.google.com/dataproc/docs/guides/dataproc-job-output#spark_driver_logs
                                                    """)
            
        num_executors_per_node_cores = yarn_cpu_vcores / spark_executor_cores
        num_executors_per_node = math.floor(num_executors_per_node_cores)
            
        
        spark_onheap_memory = solve_equation(yarn_memory_mb, spark_executor_memory_overhead_percent, spark_offheap_memory, num_executors_per_node)
            
        spark_num_executors = max(0,num_executors_per_node*num_workers) if spark_submit_deploy_mode == "client" else max(0,num_executors_per_node*num_workers) - 1
        

    return spark_executor_cores,spark_executor_memory_overhead_percent,spark_memory_fraction,spark_memory_storage_fraction,spark_offheap_memory,spark_submit_deploy_mode,num_executors_per_node,spark_onheap_memory,spark_num_executors

def node_config(node, num_workers, capacity_scheduler):
    with node:

        total_ram_per_node = st.number_input("Total RAM per node in GB", value=64)
        cores_per_node = st.number_input("Cores per node", 1, 100, value=4)
        percent_ram_for_os = st.number_input("Percent of RAM for OS", 0, 100, value=10, help="At least GiB should be reserved for OS daemons")
            
            
        yarn_cpu_vcores = cores_per_node
        reserve_core = st.radio("Reserve 1 core for OS daemon? Recommended if you are using YARN.", ["Yes", "No"])
        if capacity_scheduler == "Dominant Resource Calculator":
            oversubscribe_cpu = st.checkbox("Do you want to oversubscribe CPU?", help="""
                                            - This number can be greater than the actual vcores to oversubscribe the CPU.
                                            - It is recommended to reserve 1 core per node required for OS/Hadoop daemons
                                            - This property will take effect only when the DoinantResourceCalculator is used
                                            - Read my blog at https://medium.com/better-programming/understanding-cpu-oversubscription-in-dataproc-hadoop-95eb92e4f45d    """)
            if oversubscribe_cpu:
                st.markdown("""
                                - This property can be set/modified in the `yarn-site.xml` file. The property is `yarn.nodemanager.resource.cpu-vcores` 
                                - Learn more at https://medium.com/better-programming/understanding-cpu-oversubscription-in-dataproc-hadoop-95eb92e4f45d
                                - Oversubscribe with caution since CPU is a finite resource. 
                                """)
                yarn_cpu_vcores = st.number_input(f"YARN CPU vCores", value=cores_per_node, disabled=False,help="1 core should be reserved for OS daemons")
            else:
                yarn_cpu_vcores = st.number_input(f"YARN CPU vCores", value=cores_per_node, disabled=True,help="1 core should be reserved for OS daemons")

        yarn_cpu_vcores = yarn_cpu_vcores - 1 if reserve_core == "Yes" else yarn_cpu_vcores
        yarn_memory_gb = (1 - percent_ram_for_os/100) * total_ram_per_node
        yarn_memory_mb = convert_to_megabytes(f"{yarn_memory_gb}g")
        total_yarn_memory_mb = yarn_memory_mb * num_workers
    return cores_per_node,yarn_cpu_vcores,reserve_core,yarn_memory_mb,total_yarn_memory_mb

def cluster_configs(cluster):
    with cluster:
        container_configs = st.container(border=True)
        num_workers = container_configs.number_input("Enter number of worker nodes", 2, 100, help = "Managed Spark clusters like Dataproc require at least 2 worker nodes")
        
        scheduler_help_notes = """
                            - This property is set in the capacity-scheduler.xml file.  
                            ``` 
                                <name>yarn.scheduler.capacity.resource-calculator</name>
                                <value>org.apache.hadoop.yarn.util.resource.DominantResourceCalculator</value>
                            ```  
                            - 
                            ```
                                    <name>yarn.scheduler.capacity.resource-calculator</name>
                                    <value>org.apache.hadoop.yarn.util.resource.DefaultResourceCalculator</value>
                            ```
                            - Using the `DominantResourceCalculator` means that the capacity scheduler will consider - the available YARN memory & the YARN cpu-vcores while scheduling containers. This is especially recommended  if the workloads are CPU bound. You may observe some under-utilization of memory if the Memory to CPU ratio is high.
                            - Using the `DefaultResourceCalculator` means that the capacity scheduler will consider ONLY the available YARN memory while scheduling containers.  It leaves the responsibility of managing CPU resources to the operating system and the underlying hardware. If the workloads are CPU bound, it is recommended to use the `DominantResourceCalculator`
                            """
        capacity_scheduler = container_configs.selectbox("Capacity Scheduler", options=["Dominant Resource Calculator", "Default Resource Calculator"], help=scheduler_help_notes)
        
                                
    return num_workers,capacity_scheduler

def set_footer():
    diagrams, links = st.tabs(["Handy diagrams", "Reference links"])
    with diagrams:
        st.image("https://spark.apache.org/docs/latest/img/cluster-overview.png")

    with links:
        st.markdown(""" \
                    - https://spark.apache.org/docs/3.5.0/tuning.html)
                    """)
        
def construct_spark_submit_command(df):
    """
    Construct a spark-submit command from a DataFrame with Spark configurations.

    Parameters:
    - df: DataFrame with columns "Recommended Spark Configurations" and "Explanation".

    Returns:
    - str: Spark-submit command.
    """
    # Extract configurations from the DataFrame
    configurations = df["Recommended Spark Configurations"]

    # Construct the Spark-submit command
    spark_submit_command = "spark-submit"

    for config in configurations:
        spark_submit_command += f" --conf {config}"

    return spark_submit_command

import pandas as pd

def construct_dataproc_submit_command(df, main_class=None, jar_path=None):
    """
    Construct a Dataproc job submission command from a DataFrame with Spark configurations.

    Parameters:
    - df: DataFrame with columns "Recommended Spark Configurations" and "Explanation".
    - cluster_name: Name of the Dataproc cluster.
    - region: Region where the Dataproc cluster is located.
    - main_class: Main class for the Spark application (optional).
    - jar_path: Path to the JAR file for the Spark application (optional).

    Returns:
    - str: Dataproc job submission command.
    """
    # Extract configurations from the DataFrame
    configurations = df["Recommended Spark Configurations"]

    # Construct the Spark-submit command
    spark_submit_command = "spark-submit"

    for config in configurations:
        spark_submit_command += f" --conf {config}"

    # Construct the Dataproc job submission command
    dataproc_submit_command = f"gcloud dataproc jobs submit spark \
        --cluster <cluster_name> \
        --region <region> \
        --properties {','.join(configurations)}"
    
    dataproc_submit_command += f" --class <main_class> <jar_path>"
    return spark_submit_command, dataproc_submit_command


        
def render_memory_breakdown_visual(total_yarn_memory_mb):
        ctr_yarn_memory_mb = st.container(border=True)
        ctr_spark_executor_memory = ctr_yarn_memory_mb.container(border=True)
        ctr_shuffle_memory_fraction = st.container(border=True)
        ctr_spark_memory_fraction = st.container(border=True)
        ctr_spark_executor_memory_overhead = st.container(border=True)

        ctr_yarn_memory_mb.write(f"yarn.nodemanager.resource.memory-mb = {total_yarn_memory_mb}m")
        ctr_spark_executor_memory.write(f"{total_yarn_memory_mb}m")