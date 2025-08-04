import * as dotenv from 'dotenv';
import { ZOOM_OAUTH_ENDPOINT } from '../constants';

// Configure dotenv before any other imports that need env variables
dotenv.config();

/**
  * Retrieve token from Zoom API
  *
  * @returns {Object} { access_token, expires_in, error }
  */
export const getToken = async () => {
  try {
    const { ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET } = process.env;

    const request = await fetch(
      `${ZOOM_OAUTH_ENDPOINT}?grant_type=account_credentials&account_id=${ZOOM_ACCOUNT_ID}`,
      {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Authorization: `Basic ${Buffer.from(`${ZOOM_CLIENT_ID}:${ZOOM_CLIENT_SECRET}`).toString('base64')}`,
        }
      },
    ).then(response => response.json())
    .catch(error => console.log(error));

    const { access_token, expires_in } = await request;

    return { access_token, expires_in, error: null };
  } catch (error) {
    return { access_token: null, expires_in: null, error };
  }
};