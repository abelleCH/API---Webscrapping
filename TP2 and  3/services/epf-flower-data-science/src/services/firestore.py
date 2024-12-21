from google.cloud import firestore
from fastapi import HTTPException

def get_parameters():
    """
    Retrieves parameters from the Firestore database.
    
    If the parameters exist in the database, they are returned as a dictionary.
    If the parameters are not found, a 404 HTTPException is raised.
    
    Returns:
        dict: The parameters from the Firestore document.
        
    Raises:
        HTTPException: If the parameters are not found in Firestore.
    """
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

def update_parameters(new_params: dict):
    """
    Updates existing parameters in Firestore. If a parameter does not exist, 
    a message is returned advising to use 'add_parameters' to add it.

    Args:
        new_params (dict): Dictionary of parameters to be updated.

    Returns:
        dict: Success message and response with updated parameters or messages for missing parameters.

    Raises:
        HTTPException: If an error occurs while updating the parameters in Firestore.
    """
    try:
        db = firestore.Client()
        doc_ref = db.collection("parameters").document("parameters")
        
        doc = doc_ref.get()
        
        if doc.exists:
            current_params = doc.to_dict()
            
            response = {}
            for key, value in new_params.items():
                if key in current_params:
                    current_params[key] = value
                    #response[key] = f"Parameter '{key}' updated successfully."
                else:
                    response[key] = f"Parameter '{key}' does not exist. Please use the 'add_parameters' function to add it."

            doc_ref.set(current_params, merge=True)
            return {"message": "Parameters updated successfully.", "response": response}
        else:
            return {
                "message": "No existing parameters found. Please use the 'add_parameters' function to add new parameters."
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating parameters: {e}")


def add_parameters(params: dict):
    """
    Adds new parameters to Firestore if they do not already exist.
    
    If a parameter already exists, it returns a message indicating that the parameter needs to be updated.
    If a parameter does not exist, it will be added to the Firestore document.
    
    Args:
        params (dict): A dictionary of parameters to be added to Firestore.
        
    Returns:
        dict: A message indicating whether parameters were added or already exist.
        
    Raises:
        HTTPException: If there is an error adding the parameters to Firestore.
    """
    try:
        db = firestore.Client()
        doc_ref = db.collection("parameters").document("parameters")

        doc = doc_ref.get()
        if doc.exists:
            current_params = doc.to_dict()
        else:
            current_params = {}

        response = {}
        for key, value in params.items():
            if key in current_params:
                response[key] = f"Parameter '{key}' already exists. Please update it."
            else:
                current_params[key] = value
                response[key] = f"Parameter '{key}' added successfully."

        doc_ref.set(current_params, merge=True)

        return {"message": "Parameters processed.", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding parameters: {e}")
