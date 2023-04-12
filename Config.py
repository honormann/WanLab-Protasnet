class Config:

    def __init__(self, graph_path=None,
                 graph_mapping=None,
                 cell_line_omic=None,
                 cell_line_drug=None,
                 cell_line_fusion=None,
                 sc_cancer=None):

        self.graph_path = graph_path
        self.graph_mapping = graph_mapping
        self.cell_line_omic = cell_line_omic
        self.cell_line_drug = cell_line_drug
        self.cell_line_fusion = cell_line_fusion
        self.cell_line_fusion = cell_line_fusion
        self.sc_cancer = sc_cancer
