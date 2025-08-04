import { Type } from "@google/genai";

// Define the function declaration for the model
export const weatherFunctionDeclaration = {
  name: 'get_current_temperature',
  description: 'Gets the current temperature for a given location.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      location: {
        type: Type.STRING,
        description: 'The city name, e.g. San Francisco',
      },
    },
    required: ['location'],
  },
};

// Define the function declaration for the model
export const zoomFunctionDeclaration = {
  name: "create_zoom_meeting",
  description: "Create a zoom meeting.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      topic: {
        type: Type.STRING,
        description: "The topic of the meeting.",
      },
      start_time: {
        type: Type.STRING,
        description:
          "The meeting's start time. The start_time must be in the datetime format of YYYY-MM-DDTHH:mm:ssZ. The timezone (Z) is +08:00.",
      },
      meeting_invitees: {
        type: Type.ARRAY,
        description: "List of meeting invitees' email.",
        items: {
          type: Type.OBJECT,
          properties: {
            email: {
              type: Type.STRING,
              description: "The invitee's email.",
            },
          },
        },
      },
    },
    required: ["topic", "start_time", "meeting_invitees"],
  },
};

export const ragFunctionDeclaration = {
  name: "rag_intro_llm_prompt_engineering", // change the function name according to what data source you uploaded.
  description: "If the user's queries are all about llm prompt engineering.", // change the function description according to what data source you uploaded.
  parameters: {}
};