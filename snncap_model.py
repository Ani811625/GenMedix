import torch
import torch.nn as nn

class BiologicalCapsuleLayer(nn.Module):
    """
    Groups flat tabular medical data into multi-dimensional relational vectors (Capsules).
    """
    def __init__(self, clinical_input_dim=3, genomic_input_dim=2, vector_dim=8):
        super(BiologicalCapsuleLayer, self).__init__()
        
        self.vector_dim = vector_dim
        
        # 1. Renal / Pharmacokinetic Feature Extractor
        # Inputs: Age, Weight, Serum Creatinine
        self.renal_extractor = nn.Sequential(
            nn.Linear(clinical_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, vector_dim)
        )
        
        # 2. Genomic & SNP Feature Extractor
        # Inputs: HLA-A*32:01, agr Group II (and TCF7L2 for diabetes)
        self.genomic_extractor = nn.Sequential(
            nn.Linear(genomic_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, vector_dim)
        )

    def squash(self, tensor, dim=-1):
        """
        The non-linear activation function for capsules. 
        Ensures the vector length is between 0 and 1, preserving feature orientation.
        """
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

    def forward(self, clinical_data, genomic_data):
        # Extract features into raw vectors
        renal_raw = self.renal_extractor(clinical_data)
        genomic_raw = self.genomic_extractor(genomic_data)
        
        # Apply squash activation to convert them into true Capsules
        renal_capsule = self.squash(renal_raw)
        genomic_capsule = self.squash(genomic_raw)
        
        # Stack capsules together. Shape: (Batch_Size, 2_Capsules, Vector_Dim)
        patient_representation = torch.stack([renal_capsule, genomic_capsule], dim=1)
        
        return patient_representation

class GenMedixSiameseNetwork(nn.Module):
    """
    The Twin Scanner: Compares two patients using the Biological Capsules 
    and calculates how identical their metabolic and genetic profiles are.
    """
    def __init__(self, clinical_dim=3, genomic_dim=2, vector_dim=8):
        super(GenMedixSiameseNetwork, self).__init__()
        
        # We bring in the Capsule Layer we built in Step 1
        self.capsule_extractor = BiologicalCapsuleLayer(clinical_dim, genomic_dim, vector_dim)

    def forward_once(self, clinical_data, genomic_data):
        """
        Runs a single patient through the Capsule Network to get their multi-dimensional coordinates.
        """
        # This outputs the patient's unique biological vector map
        return self.capsule_extractor(clinical_data, genomic_data)

    def forward(self, patient_A_clinical, patient_A_genomic, patient_B_clinical, patient_B_genomic):
        """
        Runs both patients through the twin networks and compares them.
        """
        # 1. Process Patient A (The New Patient)
        vector_map_A = self.forward_once(patient_A_clinical, patient_A_genomic)
        
        # 2. Process Patient B (The Historical Database Patient)
        vector_map_B = self.forward_once(patient_B_clinical, patient_B_genomic)
        
        # 3. Calculate the distance between them
        # We flatten the capsules and calculate the Euclidean distance (straight-line distance)
        flat_A = vector_map_A.view(vector_map_A.size(0), -1)
        flat_B = vector_map_B.view(vector_map_B.size(0), -1)
        
        # Pairwise distance calculates how far apart Patient A and B are in the network's "brain"
        distance = torch.nn.functional.pairwise_distance(flat_A, flat_B)
        
        return distance, vector_map_A, vector_map_B

    def calculate_similarity_percentage(self, distance):
        """
        Converts the raw mathematical distance into an easy-to-read 0% to 100% score for the doctors.
        """
        # A distance of 0 means 100% identical. 
        # The torch.exp(-distance) safely converts the gap into a percentage.
        similarity = torch.exp(-distance) * 100.0
        return similarity
class ContrastiveLoss(nn.Module):
    """
    The Teacher: Forces the network to pull matching patients together 
    and push different patients apart in the multi-dimensional space.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        # The 'margin' is the minimum safe distance we want between non-matching patients.
        self.margin = margin

    def forward(self, distance, label):
        """
        distance: The mathematical gap between Patient A and B (calculated in Step 2).
        label: 1 if they have the same optimal dose (Twins), 0 if they need different doses (Strangers).
        """
        
        # 1. If they are TWINS (label = 1): 
        # We want the distance to be exactly 0. Any distance greater than 0 is an error (loss).
        loss_twin = label * torch.pow(distance, 2)
        
        # 2. If they are STRANGERS (label = 0):
        # We want their distance to be AT LEAST our margin (2.0). 
        # If they are closer than the margin, the network gets penalized.
        loss_stranger = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        # The final error is the average of these calculations. 
        # The network will adjust its weights to make this number as close to 0 as possible.
        loss_contrastive = torch.mean(loss_twin + loss_stranger)
        
        return loss_contrastive
    
class SNNCapNetwork(nn.Module):
    """
    The full Multi-Task Architecture: Combines the Biological Capsules with a 
    Regression Head to output the exact continuous dosage (mg/day).
    """
    def __init__(self, input_dim=5, capsule_dim=8):
        super(SNNCapNetwork, self).__init__()
        
        # 1. Bring in your existing Capsule Layer
        # Clinical = Age, Weight, CrCl (3) | Genomic = HLA, agr_Mutation (2)
        self.capsule_layer = BiologicalCapsuleLayer(
            clinical_input_dim=3, 
            genomic_input_dim=2, 
            vector_dim=capsule_dim
        )
        
        # 2. The Dosage Regression Head
        # Flattens the two 8D capsules (2 * 8 = 16) and calculates the dose
        self.regression_head = nn.Sequential(
            nn.Linear(2 * capsule_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Outputs a single continuous dose value
        )

    def forward(self, x):
        """
        x: The flat 5-dimensional tensor from your X_master matrix
        """
        # Automatically split the flat input tensor into the two required biological branches
        clinical_data = x[:, :3]   # First 3 columns (Age, Weight, CrCl)
        genomic_data = x[:, 3:]    # Last 2 columns (HLA, agr_Mutation)
        
        # Pass through the capsule layer
        capsules = self.capsule_layer(clinical_data, genomic_data)
        
        # Flatten the capsules to feed into the dosage calculator
        flat_capsules = capsules.view(capsules.size(0), -1)
        
        # Calculate the final continuous dosage
        predicted_dose = self.regression_head(flat_capsules)
        
        # Return both the dose prediction and the raw capsules (for metric tracking)
        return predicted_dose, capsules