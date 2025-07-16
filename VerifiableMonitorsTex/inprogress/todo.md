- ~~R2 - Reduction to MDPs: Note that we are looking for a policy in a colored MDP (See Section 3.1), which is NP-hard. This contrasts with looking for policies in (standard) MDPs, which can be computed in polynomial time. The correctness of our approach depends on the use of colored MDPs, **we will add an example to clarify this at the end of Section 3.3.**~~

- ~~R2 - Explanation of the equivalence query. The equivalence query is realized by our novel verification procedure (see page 13). **We will clarify this connection more explicitly.** This also motivates the title. By establishing this first verification algorithm, we can implement the necessary EQs to learn correct monitors. ~~

- R3 - We only use conformance checking to accelerate the learning of monitors. We can ensure completeness by our novel equivalence oracle, which uses our verification routine (Section 3). Using equivalence oracles is necessary to be correct, as is highlighted in our Experiment (Figure 9). **We will clarify at the end of Section 4 how the conformance checking and verification are combined in our method.**

- ~~Page 5, line 3 after Definition 1: There is a typo in the subscript for the dots. Also shouldn't it be S \times (Act \times S)^* \cup (S \times Act)^{\omega} instead of (S \times Act)^* \times S, since infinite paths as well as the empty path needs to be included?~~

- ~~Page 5, example 1: Do you mean q_d in place of q_c?~~

- ~~Page 6, Definition 5: Shouldn't the subscript in the notation for missed alarms be \lambda_u and not \lambda_s? (Also everywhere else in the paper?)~~

- Page 8, lines 2,3 : Wouldn't it be better to use some notation other than F for alarm states, i.e., non-accepting states in the DFA, since accepting states are also (by default) denoted by F?

- ~~Page 8, definition 10 and lemma 1: Shouldn't \lambda_s be \lambda_u? We are looking for unsafe traces that are not accepted by the monitor and a trace is unsafe iff its risk is strictly above \lambda_u. For the same reason, shouldn't the \ge in definition 10 instead be a >?~~

- ~~Page 11, Theorem 3: Shouldn't we be looking for a trace whose risk value is below \lambda_S and is accepted by the monitor?~~

- ~~R1: Regarding footnote 6. A: Let us first consider an example: Take the colored MDP in Fig. 6, where the color-consistent policy with the lowest probability of reaching the alarm state is as follows: take the icy action at step 1, take the dry action at step 2, and the z_end action at step 3. This policy never reaches an end state and is thus not a trace in the monitor, but it minimizes reaching the alarm state. More generally, models from Def. 12 allow for schedulers which induce end components that do not entail traces in the monitor (or its complement). *We will clarify footnote 6.*~~
- ~~Page 12, footnote 6: This is not entirely clear to me. Shouldn't the policy choose one action for each state and isn't the only available action for the states at the end, the z_end action?~~

- ~~Page 1, section 1, line 15: Challenges (3,4) -> and Challenges (3,4)~~

- ~~Page 2, monitoring with HMMs subsection, line 2: Is it better to state "performs inference on the HMM modelling the system"?~~

- ~~Page 2, last two lines: 13/22 for this trace (see example 2) -> (13/22 for this trace, see example 2).~~

- ~~Page 3, Fig 3a), line 3: of the road -> off the road~~

- ~~Page 3, line 10: focus -> focus on~~

- ~~Page 4, contribution subsection, line 7: the hardness -> hardness~~

- ~~Page 4, section 2, line 7: recursively: -> recursively as:~~

- ~~Page 4, section 2, line 9: accepts w \in L(A) -> accepts w if w \in L(A)~~

- ~~Page 5, line 6 after Definition 1: at most length h are -> at most length h is. Similarly in the next line.~~

- ~~Page 5, paragraph 3, line 5: associate -> associated~~

- ~~Page 6, Problem statements: The subscripts for mA and fA do not use the correct \lambda term.~~

- ~~Page 6, penultimate line: construction of -> construction to~~

- ~~Page 7, two lines above corollary 2: both include -> include both~~

- ~~Page 8, Fig 5: The probability of going from (1,(d,1)) to (2,(i,2)) must be 9/10.~~

- ~~Page 8, example 4, line 5: any product states -> any product states of the form~~

- ~~Page 8, definition 10, line 2: The phrase "decide if " needs to be added at the end~~

- ~~Page 9, line 2: eliminate -> eliminating~~

- ~~Page 9, example 5, line 6: It is not clear which state you are referring to. Do you mean to say , "since r((i,3)) = 0"?. This example was not clear to me before reading Definition 11.~~

- ~~Page 10, line 1: from Fig. 6 -> in Fig. 6~~

- R2: Definition 12 and 13
A: These definitions require an unrolled HMM definition (which is Def. 11). *We will clarify the wording to highlight what aspects are relevant.*

- ~~Page 10, Definition 12, line 2: S \ {t_{ignr}} -> S \ {(h+1,t_{ignr})}. Also in the definition of P', the RHS must have P instead of P'.~~

- ~~Page 10, Definition 13, line 2: \tau \in Z^*. -> \tau \in Z^*, . Also the star superscript of Z must be changed.~~

- ~~Page 11, Fig 7b), line 1: bisimular -> bisimilar ??~~

- ~~Page 11, section 3.4, line 2: exactly the horizon. -> exactly equal to the horizon.~~

- ~~Page 11, line 3 after Theorem 2: can instead -> can also instead~~

- Page 12, section 4, line 1: It might be better to phrase the starting of the sentence in another manner, for instance, as "We now consider Problem 3, i.e., learning correct monitors. We accomplish this by combining ..."

- ~~Page 12, Definition 15, line 2: The superscript * must be changed for Z. Also the dot after Z^* must be replaced by a , and the "Such" must be in lowercase.~~

- ~~Page 12, Definition 15, line 3: the missed -> a missed~~

- ~~Page 13, line 4 from the bottom: represents assignment -> represents an assignment~~

- ~~Page 14, last line before Section 6: on the HMM -> using the HMM~~

- ~~Page 14, section 6, line 2: (Sec. 4) -> (Sec. 4) algorithms~~

- ~~Page 14, fourth line from the bottom: C++ -> C++,~~

- ~~Page 15, last line: \lambda_u cannot be 0.1, while \lambda_s is 0.3, right?~~

- ~~Page 17, Fig 9, last line: out-of-5memory -> out-of-5memory~~

- ~~Page 18, line 2: work those directions -> work in those directions~~

- ~~p 5, l6: s_0.a_0..._1 -> there is a problem here~~
- ~~p5, l15: sigma-algebra associate -> sigma-algebra associated~~
- ~~p5, Example 1: the trace q_c . q_i . q_i should be q_d. q_i . q_i I guess~~
- ~~p7, Theorem 1: I don't think the notation Pr^{M^C}_\sigma has been introduced~~
- ~~p 8, Definition 9: in the HMM M\times A, Z is missing.~~
- ~~p 10, Definition 13 :a trace tau in Z^*. A trace -> a trace tau in Z^*, a trace~~
- ~~p11, Lemma 3: , such that -> something is wrong here~~
- ~~p 11, Lemma 3: to build the HMM of the lemma, we need also alarm states F at least.~~
- ~~p11, Lemma 3: please highlight more clearly that you obtain trace-consistent policies when obtaining a color-consistent policy.~~
- ~~p12, Definition 15: . Such that -> such that~~
- ~~p13, l14: has a constant cost -> as a constant cost.~~