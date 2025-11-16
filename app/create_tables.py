from app.database import engine
from app.models import Base

# Créer toutes les tables
Base.metadata.create_all(bind=engine)

print("✅ Table 'plant_diseases' créée avec succès!")