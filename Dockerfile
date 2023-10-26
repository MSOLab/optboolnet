FROM colomoto/colomoto-docker:next

USER root
RUN conda install -y \
        -c MSO_Lab \
        -c Gurobi -c conda-forge -c colomoto \
        optboolnet \
        colomoto_jupyter pyomo gurobi \
    && conda clean -y --all && rm -rf /opt/conda/pkgs

ENV GRB_LICENSE_FILE=/notebook/gurobi.lic

USER user
