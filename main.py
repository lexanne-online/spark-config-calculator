import math
import streamlit as st
import pandas as pd
from scipy.optimize import fsolve

# Function to convert size string to megabytes

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
            
    spark_onheap_memory = int(fsolve(lambda z: z + max(384, p * z) - (x / y - m), initial_guess)[0])
    return spark_onheap_memory

def main():
    st.set_page_config(
    page_title="Spark Configuration Tool",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
    # Set the layout for the page
    #st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: white;'>Spark Configuration Calculator</h1>", unsafe_allow_html=True)
    columns = st.columns(8)

    with columns[2]:
        st.write("""<div style="width:100%;text-align:center;"><a href="https://www.jeromerajan.com" style="float:center"><img src="https://cdn0.iconfinder.com/data/icons/england-13/504/sherlock-holmes-detective-inspector-man-512.png" width="22px"></img></a></div>""", unsafe_allow_html=True)

    
    with columns[3]:
        st.write("""<div style="width:100%;text-align:center;"><a href="https://linkedin.com/in/jeromerajan" style="float:center"><img src="https://cdn2.iconfinder.com/data/icons/social-media-applications/64/social_media_applications_14-linkedin-512.png" width="22px"></img></a></div>""", unsafe_allow_html=True)
        
    with columns[4]:
        st.write("""<div style="width:100%;text-align:center;"><a href="https://medium.com/@datasherlock" style="float:center"><img src="https://cdn2.iconfinder.com/data/icons/social-icons-33/128/Medium-512.png" width="22px"></img></a></div>""", unsafe_allow_html=True)
        
    with columns[5]:
        st.write("""<div style="width:100%;text-align:center;"><a href="https://github.com/datasherlock" style="float:center"><img src="https://cdn3.iconfinder.com/data/icons/social-media-2169/24/social_media_social_media_logo_github_2-512.png" width="22px"></img></a></div>""", unsafe_allow_html=True)

    st.markdown("""---""")
    
    left, right = st.columns(2)

    with left:
        cluster, node, executor = st.tabs(["Cluster Configuration", "Node Configuration", "Executor Configuration"])

        with cluster:
            num_workers = st.number_input("Enter number of workers", 2, 100)
            capacity_scheduler = st.selectbox("Capacity Scheduler", options=["Dominant Resource Calculator", "Default Resource Calculator"])
            if capacity_scheduler == "Dominant Resource Calculator":
                st.markdown("""
                            - This property is set in the capacity-scheduler.xml file.  
                            ``` 
                                <name>yarn.scheduler.capacity.resource-calculator</name>
                                <value>org.apache.hadoop.yarn.util.resource.DominantResourceCalculator</value>
                            ```  
                            - Using the `DominantResourceCalculator` means that the capacity scheduler will consider - the available YARN memory & the YARN cpu-vcores while scheduling containers. This is especially recommended  if the workloads are CPU bound. You may observe some under-utilization of memory if the Memory to CPU ratio is high.
                            """)
            else:
                st.markdown("""
                            - This property is set in the capacity-scheduler.xml file.  
                            - ```<property>
                                    <name>yarn.scheduler.capacity.resource-calculator</name>
                                    <value>org.apache.hadoop.yarn.util.resource.DefaultResourceCalculator</value>
                                </property>```
                            - Using the `DefaultResourceCalculator` means that the capacity scheduler will consider ONLY the available YARN memory while scheduling containers.  It leaves the responsibility of managing CPU resources to the operating system and the underlying hardware. If the workloads are CPU bound, it is recommended to use the `DominantResourceCalculator`
                            """)


        with node:
            #st.header("Node Configuration")
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
            yarn_memory_mb = (1 - percent_ram_for_os/100) * total_ram_per_node * 1024
            total_yarn_memory_mb = yarn_memory_mb * num_workers

        


        with executor:
           # st.header("Executor Configuration")
            #spark_executor_memory = convert_to_megabytes(st.text_input("spark.executor.memory", value="4g"))
            spark_executor_cores = st.number_input("spark.executor.cores (cores per executor)", 1, yarn_cpu_vcores, \
                                                   help=""" 
                                                   - This is the number of parallel tasks in each executor (Recommended to keep under 5 for Yarn)  
                                                   - This defaults to 1 when using YARN and defaults to all available cores in a node when using standalone
                                                   """, value=min(yarn_cpu_vcores, 5))
            spark_executor_memory_overhead_percent = (st.number_input("spark.executor.memoryOverheadFactor %", 0, 100, value = 6, help="""6%-10% ideally. Allocate more as executor size increases. This is done as non-JVM tasks need more non-JVM heap space and such tasks commonly fail with "Memory Overhead Exceeded" errors.""") / 100)
            spark_memory_fraction = st.number_input("spark.memory.fraction", value=0.6, help="The default is 0.6 in Spark 3.x. Recommended to use the default")
            spark_memory_storage_fraction = st.number_input("spark.memory.storageFraction", value=0.5, help="The default is 0.5 in Spark 3.x. Recommended to use the default")
            #spark_yarn_executor_memoryOverhead = round(max(384, spark_executor_memory_overhead_percent * spark_executor_memory), 3)
            #st.text(f"spark.yarn.executor.memoryOverhead (MB) = {spark_yarn_executor_memoryOverhead}")
            spark_offheap_memory = st.number_input("spark.memory.offHeap.size (in MB)", value=0, help="The absolute amount of memory which can be used for off-heap allocation") 

            num_executors_per_node_cores = yarn_cpu_vcores / spark_executor_cores
            #num_executors_per_node_memory = yarn_memory_mb/ (spark_offheap_memory  + spark_yarn_executor_memoryOverhead + spark_executor_memory)
            #num_executors_per_node = min(num_executors_per_node_cores, num_executors_per_node_memory) if capacity_scheduler == "Dominant Resource Calculator" else num_executors_per_node_memory
            num_executors_per_node = num_executors_per_node_cores
            
        
            spark_onheap_memory = solve_equation(yarn_memory_mb, spark_executor_memory_overhead_percent, spark_offheap_memory, num_executors_per_node)
            
            spark_num_executors = max(0,math.floor(num_executors_per_node*num_workers))

            #spark_onheap_memory = (yarn_memory_mb - num_executors_per_node * spark_offheap_memory) / (num_executors_per_node + spark_yarn_executor_memoryOverhead)
            # spark_onheap_memory = (yarn_memory_mb - num_executors_per_node_cores*spark_offheap_memory) / (num_executors_per_node_cores * (1 + spark_executor_memory_overhead_percent)) 
            # print(yarn_memory_mb, num_executors_per_node_cores, spark_offheap_memory, spark_onheap_memory)
        st.markdown("""---""")
    
        
    with right:
        results, memory_breakdown, revised = st.tabs(["Recommended configurations", "On-heap Memory Breakdown", "Revised Configurations (DefaultResourceCalculator)"])
        
        with results:
            df = pd.DataFrame({
                'Recommended Spark Configurations': [
            f"spark.executor.memory = {spark_onheap_memory}m",
            f"spark.executor.cores= {spark_executor_cores}",
            f"spark.memory.fraction= {spark_memory_fraction}",
            f"spark.memory.storageFraction= {spark_memory_storage_fraction}",
            f"spark.memory.offHeap.size= {spark_offheap_memory}",
            f"spark.executor.memoryOverheadFactor= {spark_executor_memory_overhead_percent}",
            f"spark.dynamicAllocation.initialExecutors = {spark_num_executors}",
            f"spark.dynamicAllocation.enabled=true"
            ], 'Explanation' : [
            f"Amount of memory to use per executor process",
            "The number of cores to use on each executor in Yarn.  \n  In standalone mode, this will equal all the cores in a node",
            f"Fraction of (heap space - 300MB) used for execution and storage. The lower this is, the more frequently spills and cached data eviction occur. The purpose of this config is to set aside memory for internal metadata, user data structures, and imprecise size estimation in the case of sparse, unusually large records.",
            f"Amount of storage memory immune to eviction, expressed as a fraction of the size of the region set aside by spark.memory.fraction. The higher this is, the less working memory may be available to execution and tasks may spill to disk more often",
            f"The absolute amount of memory which can be used for off-heap allocation, in bytes unless otherwise specified. This setting has no impact on heap memory usage, so if your executors' total memory consumption must fit within some hard limit then be sure to shrink your JVM heap size accordingly. ",
            f"Fraction of executor memory to be allocated as additional non-heap memory per executor process. This is memory that accounts for things like VM overheads, interned strings, other native overheads, etc. This tends to grow with the container size.",
            f"Initial number of executors to run if dynamic allocation is enabled.",
            f"Whether to use dynamic resource allocation, which scales the number of executors registered with this application up and down based on the workload."
            ]
            })
            #st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
            st.write(df)

            storage_memory = round((spark_onheap_memory - 300) * spark_memory_storage_fraction * spark_memory_fraction,2)
            execution_memory = round((spark_onheap_memory - 300) * spark_memory_fraction * (1 - spark_memory_storage_fraction),2)
            user_memory = round(spark_onheap_memory - (storage_memory + execution_memory) - 300,2)

            total_memory_utilised = round(spark_onheap_memory * num_executors_per_node * num_workers,2)
            total_cores_utilised = spark_executor_cores * num_executors_per_node * num_workers
            total_physical_cores = (cores_per_node - 1)*num_workers if reserve_core == "Yes" else cores_per_node * num_workers
        
            
            _ , memory, cpu = st.columns(3)
            memory.metric("Memory Utilisation", f"{total_memory_utilised}m" , f"{round((total_memory_utilised - total_yarn_memory_mb)/total_yarn_memory_mb*100,2)}%" )
            cpu.metric("CPU Utilisation", f"{total_cores_utilised} vcores" , f"{round((total_cores_utilised - total_physical_cores)/total_physical_cores*100,2)}%" )


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


            _ , memory, cpu = st.columns(3)
            memory.metric("Memory Utilisation", f"{total_memory_utilised}m" , f"{round((total_memory_utilised - total_yarn_memory_mb)/total_yarn_memory_mb*100,2)}%" )
            cpu.metric("CPU Utilisation", f"{total_cores_utilised} vcores" , f"{round((total_cores_utilised - total_physical_cores)/total_physical_cores*100,2)}%" )

            
                
        with revised:
            if capacity_scheduler == "Default Resource Calculator":
                revised_spark_executor_memory = convert_to_megabytes(st.text_input("Revised spark.executor.memory", value=str(spark_onheap_memory) + 'm'))
                num_executors_per_node_memory = yarn_memory_mb/ (spark_offheap_memory  + spark_executor_memory_overhead_percent*revised_spark_executor_memory + revised_spark_executor_memory)
                revised_num_executors = max(0,math.floor(num_executors_per_node_memory*num_workers))



                df_revised = pd.DataFrame({
                        'Recommended Spark Configurations': [
                    f"spark.executor.memory = {int(revised_spark_executor_memory)}m",
                    f"spark.executor.cores= {spark_executor_cores}",
                    f"spark.memory.fraction= {spark_memory_fraction}",
                    f"spark.memory.storageFraction= {spark_memory_storage_fraction}",
                    f"spark.memory.offHeap.size= {spark_offheap_memory}",
                    f"spark.executor.memoryOverheadFactor= {spark_executor_memory_overhead_percent}",
                    f"spark.dynamicAllocation.initialExecutors = {revised_num_executors}",
                    f"spark.dynamicAllocation.enabled=true"
                    ], 'Explanation' : [
                    f"Amount of memory to use per executor process",
                    "The number of cores to use on each executor in Yarn.  \n  In standalone mode, this will equal all the cores in a node",
                    f"Fraction of (heap space - 300MB) used for execution and storage. The lower this is, the more frequently spills and cached data eviction occur. The purpose of this config is to set aside memory for internal metadata, user data structures, and imprecise size estimation in the case of sparse, unusually large records.",
                    f"Amount of storage memory immune to eviction, expressed as a fraction of the size of the region set aside by spark.memory.fraction. The higher this is, the less working memory may be available to execution and tasks may spill to disk more often",
                    f"The absolute amount of memory which can be used for off-heap allocation, in bytes unless otherwise specified. This setting has no impact on heap memory usage, so if your executors' total memory consumption must fit within some hard limit then be sure to shrink your JVM heap size accordingly. ",
                    f"Fraction of executor memory to be allocated as additional non-heap memory per executor process. This is memory that accounts for things like VM overheads, interned strings, other native overheads, etc. This tends to grow with the container size.",
                    f"Initial number of executors to run if dynamic allocation is enabled.",
                    f"Whether to use dynamic resource allocation, which scales the number of executors registered with this application up and down based on the workload."
                    ]
                    })
                st.write(df_revised)

                revised_memory_utilised = round(revised_spark_executor_memory * revised_num_executors ,2)
                revised_cores_utilised = spark_executor_cores * revised_num_executors 
                physical_cores = cores_per_node*num_workers
                
                _ , memory, cpu = st.columns(3)
                memory.metric("Memory Utilisation", f"{revised_memory_utilised}m" , f"{round((revised_memory_utilised - total_yarn_memory_mb)/total_yarn_memory_mb*100,2)}%" )
                cpu.metric("CPU Utilisation", f"{revised_cores_utilised} vcores" , f"{round((revised_cores_utilised - total_physical_cores)/total_physical_cores*100,2)}%", help="Reserving a core per node for OS may show under-utilisation here" )

                
            else:
                st.write("Not applicable for DominantResourceCalculator")  
      
    

        if capacity_scheduler == "Default Resource Calculator" :
            st.markdown("""
                        - These configurations ensure that all the memory is used for executors with the assumption that each executor will consume the requested cores. 
                        - But since you are using `DefaultResourceCalculator`, the container allocation can be done using memory alone leaving CPU sharing to be managed by the OS
                        - You can reduce the `spark.executor.memory` to a smaller value that's more suited to your workload and run another applications concurrently.
                        - This way, Yarn will schedule more executors requested by the next application as long as there is memory available.
                        - Click the revised configurations tab above to tune the executor memory and see the revised configurations based on different values of `spark.executor.memory`
                        """) 
    
    

if __name__ == "__main__":
    main()
