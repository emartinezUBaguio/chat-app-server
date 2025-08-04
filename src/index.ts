import express, { Request, Response } from 'express';
import cors from 'cors';
import multer from 'multer';
import { handleFileUpload, handleChat, handleGenerate, handleFineTuned } from './services';
import * as dotenv from 'dotenv';

// Configure dotenv before any other imports that need env variables
dotenv.config();

// --- SERVER CONFIGURATION ---
const app = express();
const port = process.env.PORT || 3000;

// --- MIDDLEWARE ---
app.use(express.json());
app.use(cors());
app.use(express.urlencoded({ extended: true }));

// --- FILE UPLOAD SETUP ---
// We'll use memory storage to handle the file buffer directly.
// For production, consider using diskStorage or a cloud storage solution.
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// --- API ROUTES ---

/**
 * @route   POST /api/upload
 * @desc    Uploads a text file, processes it, and stores its embeddings.
 * @access  Public
 */
app.post('/api/upload', upload.single('file'), async (req: Request, res: Response) => {
    try {
        if (!req.file) {
            return res.status(400).json({ message: 'No file uploaded.' });
        }

        const file = req.file;
        const documentId = await handleFileUpload(file.buffer, file.originalname);

        res.status(201).json({
            message: 'File uploaded and processed successfully.',
            documentId: documentId
        });
    } catch (error) {
        console.error('Error during file upload:', error);
        res.status(500).json({ message: 'An error occurred during file processing.' });
    }
});

/**
 * @route   POST /api/chat
 * @desc    Handles a chat query using RAG with the stored documents.
 * @access  Public
 */
app.post('/api/chat', async (req: Request, res: Response) => {
    try {
        const { query, history } = req.body;

        if (!query) {
            return res.status(400).json({ message: 'Query is required.' });
        }

        const response = await handleChat(query, history || []);

        res.status(200).json({
            message: 'Response generated successfully.',
            response: response
        });
    } catch (error) {
        console.error('Error during chat:', error);
        res.status(500).json({ message: 'An error occurred while generating a response.' });
    }
});

/**
 * @route   POST /api/generate
 * @desc    Handles a chat query using function calling.
 * @access  Public
 */
app.post('/api/generate', async (req: Request, res: Response) => {
    try {
        const { query, history } = req.body;

        if (!query) {
            return res.status(400).json({ message: 'Query is required.' });
        }

        const response = await handleGenerate(query, history || []);

        res.status(200).json({
            message: 'Response generated successfully.',
            response: response
        });
    } catch (error) {
        console.error('Error during chat:', error);
        res.status(500).json({ message: 'An error occurred while generating a response.' });
    }
});

/**
 * @route   POST /api/fine-tuned
 * @desc    Handles a chat query using fine-tuned model.
 * @access  Public
 */
app.post('/api/fine-tuned', async (req: Request, res: Response) => {
    try {
        const { query, history } = req.body;

        if (!query) {
            return res.status(400).json({ message: 'Query is required.' });
        }

        const response = await handleFineTuned(query, history || []);

        res.status(200).json({
            message: 'Response generated successfully.',
            response: response
        });
    } catch (error) {
        console.error('Error during chat:', error);
        res.status(500).json({ message: 'An error occurred while generating a response.' });
    }
});

// --- START SERVER ---
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});

export default app;
