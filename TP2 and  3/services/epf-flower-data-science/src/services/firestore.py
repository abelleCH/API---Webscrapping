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
    Updates the parameters in Firestore with new values.
    
    This function checks if the parameters already exist. If they do, the existing parameters 
    are updated with the new ones. If they don't exist, the new parameters are added to Firestore.
    
    Args:
        new_params (dict): A dictionary containing the new parameters to be updated.
        
    Returns:
        dict: A success message and the updated parameters.
        
    Raises:
        HTTPException: If there is an error updating the parameters in Firestore.
    """
    try:
        db = firestore.Client()
        doc_ref = db.collection("parameters").document("parameters")
        
        doc = doc_ref.get()
        if doc.exists:
            current_params = doc.to_dict().get("params", {})
        else:
            current_params = {}

        # Update parameters
        current_params.update(new_params)

        # Save the updated parameters back to Firestore
        doc_ref.set({"params": current_params}, merge=True)

        return {"message": "Parameters updated successfully.", "params": current_params}
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

        # Retrieve existing parameters
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
                # Add new parameter if not exists
                current_params[key] = value
                response[key] = f"Parameter '{key}' added successfully."

        # Save the parameters to Firestore
        doc_ref.set(current_params, merge=True)

        return {"message": "Parameters processed.", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding parameters: {e}")
