# Rule tags

Every refusal is printed as `error[<rule>]: <message>`. The rule is the
stable part; the message is for a person. Each has a fixture in
`tests/reject/<rule>.lx` that reaches it, except where noted.

| Rule | Raised by | Meaning |
|---|---|---|
| `syntax` | lexer, parser | The text is not a `.lx` program |
| `unbound` | elaboration | A name that is not a constant, parameter, binding or operation |
| `offset` | elaboration | An offset or extent that is not affine in the indices, or does not divide |
| `shape` | elaboration, checker | Shapes that do not fit: broadcasts, matmul and mma dimensions, quantised groups |
| `carry` | elaboration, checker | A loop that carries more than one tile, never yields, or yields another type |
| `grid` | elaboration | `grid` inside a loop, more than two axes, or a grid twice |
| `chunk` | elaboration | A schedule's `chunk` that does not divide, or parameters cut differently |
| `no-schedule` | parser, CLI | No schedule for the target (no fixture: a file with none for either target, `tests/reject/no-schedule.lx.txt`) |
| `use-after-move` | checker | A tile used after it was consumed |
| `moved-while-borrowed` | checker | One op that moves a tile and also uses it |
| `leak` | checker | A tile never consumed |
| `consume-outer` | checker | A loop body consuming a tile defined outside it |
| `scope` | checker | A variable out of scope (no fixture: the elaborator cannot produce one; kept as the checker's defence) |
| `type` | checker | A fragment where a numbered tile is wanted, arithmetic on I8, a read-only store |
| `narrowing` | checker | An implicit conversion to a smaller dtype |
| `bounds` | checker | A view outside its parameter on some iteration |
| `budget` | checker, lowering | Threadgroup memory past the target's limit, or past a static declaration's |
| `lowering` | lowering | A program the emitters cannot lower (a warp grid that does not match the threads, ...) |
| `interp` | interpreter | A `--run` whose tensors do not fit the interpreter's memory, or a value it cannot find (the latter cannot happen after the checker) |
| `usage` | CLI | A constant that is not `name=number` |
| `io` | CLI | A file that cannot be read or written |
