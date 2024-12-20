from google.cloud import firestore
from fastapi import HTTPException

def get_parameters():
    try:
        db = firestore.Client()
        doc_ref = db.collection("parameters").document("parameters")
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Parameters not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving parameters: {e}")
