# ShakespeareRNN

In Andrej Karpathy's excellent blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) he describes training an LSTM model on the collected works of Shakespeare to see what it produces. That sounded like fun so I thought I'd give it a try.

This is a character-level recurrent network, so it has no pre-conceived knowledge of English words and can only output one character at a time. At the beginning of training, all that comes out of the net is random gibberish, but as training progesses, language constructs begin to emerge--from groupings of letters to real words to sentences. Interestingly, the net isn't just learning to copy the training dataset--the sentences it outputs are not found in Shakespeare's work at all. Here is a sample of its output:

```   
        Enter LA PUCENTA'S councillow

KING JOHN. 'Tis not evenly and nan hand
In the honours of Suffolk's imprisonment;
Four duty of Count, sing in't.                                 Exit ambitious.
ANTONY. That hour, beshrew my boy, sweat? Women see,
the a heels will be a present solemnity;
If the time is offer his own princely case!
SHALLOW. I fear you know me there, not the prey piece- lest you
had her nothing and come where think to be my bed with a together.
SHALLOW. Alike, you that stand in me
Appointed upon to guard, take it, our old parator
Up together, I do noise me, I thunder.
KING RICHARD. O, tell me, my lord,
I'll speak you they but put and see their brining
That so wisely with bating, and
Were the hisbond's ears of me. Sound love.
I am spriel'd, this in your counterfeilence
to run thee to thine offend, or the Eagle
Which opinion'd I believe thee with thee for;
No faithful dhoulds, if skill would awake
From form, and so young manner vow,
If you priviled pleady in delight-
Though the more we phrase, lethare at the tenal
Cannot with what breed son of Caliban
And turn and devoted by mine eyes.
Ghost put up in the himself; 'It comes too it. Heaven shallow
I reason.
NORFOL. Make you dispatch'd, my commanded gold,
That all of paids does refuse, his
Soothsails can maich?
CHIEF JUSTICE. To remember me; why shall I play the Duke
Shall take lost.
BRUTUS. Nothing do good news.
But in your poor pen.
LEONTES. And good power; thy thoughts shall give
Young-gentle shame wherein you gladly
Enough of sound, Master Shence.                                                  Exeunt
```

### LSTM Model

For this learning task, I used a 3-layer Long Short Term Memory (LSTM) model with a hidden size of 512 nodes at each layer. Since the Shakespeare dataset is relatively small, overfitting can be an issue, but my main consideration for the model size was keeping the training time reasonable. Larger models would most likely perform better, albeit with diminishing returns. I trained with the Adam optimizer with a learning rate of 5e-3. 
