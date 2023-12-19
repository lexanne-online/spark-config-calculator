import streamlit as st
from utils import *

def main():
    set_page_header_format()
    left, right = st.columns(2)

    with left:
        cluster, node, executor = st.tabs(["Cluster Configuration", "Node Configuration", "Executor Configuration"])

        num_workers, capacity_scheduler = cluster_configs(cluster)
        cores_per_node, yarn_cpu_vcores, reserve_core, yarn_memory_mb, total_yarn_memory_mb = node_config(node, num_workers, capacity_scheduler)
        spark_executor_cores, spark_executor_memory_overhead_percent, spark_memory_fraction, spark_memory_storage_fraction, spark_offheap_memory, spark_submit_deploy_mode, num_executors_per_node, spark_onheap_memory, spark_num_executors = spark_executor_config(executor, num_workers, capacity_scheduler, yarn_cpu_vcores, yarn_memory_mb)            
        
        st.markdown("""---""")
  
        
    with right:
        results, memory_breakdown, revised = st.tabs(["Recommended configurations", "On-heap Memory Breakdown", "Revised Configurations (DefaultResourceCalculator)"])
        
        storage_memory, execution_memory, user_memory, total_memory_utilised, total_cores_utilised, total_physical_cores = recommendations(num_workers, cores_per_node, reserve_core, total_yarn_memory_mb, spark_executor_cores, spark_executor_memory_overhead_percent, spark_memory_fraction, spark_memory_storage_fraction, spark_offheap_memory, spark_submit_deploy_mode, num_executors_per_node, spark_onheap_memory, spark_num_executors, results)
        memory_breakdown_guidance(total_yarn_memory_mb, memory_breakdown, storage_memory, execution_memory, user_memory, total_memory_utilised, total_cores_utilised, total_physical_cores)
        revised_recommendations(num_workers, capacity_scheduler, cores_per_node, yarn_memory_mb, total_yarn_memory_mb, spark_executor_cores, spark_executor_memory_overhead_percent, spark_memory_fraction, spark_memory_storage_fraction, spark_offheap_memory, spark_submit_deploy_mode, spark_onheap_memory, revised, total_physical_cores)  
        if capacity_scheduler == "Default Resource Calculator" :
            st.markdown("""
                        - These configurations ensure that all the memory is used for executors with the assumption that each executor will consume the requested cores. 
                        - But since you are using `DefaultResourceCalculator`, the container allocation can be done using memory alone leaving CPU sharing to be managed by the OS
                        - You can reduce the `spark.executor.memory` to a smaller value that's more suited to your workload and run another applications concurrently.
                        - This way, Yarn will schedule more executors requested by the next application as long as there is memory available.
                        - Click the revised configurations tab above to tune the executor memory and see the revised configurations based on different values of `spark.executor.memory`
                        """) 
    
    set_footer()


if __name__ == "__main__":
    main()
