; out[i] = in[i] * factor for every i in [0, n)
define void @scale(ptr noalias %out, ptr noalias %in, i64 %n, double %factor) {
entry:
  %empty = icmp sle i64 %n, 0
  br i1 %empty, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %src = getelementptr inbounds double, ptr %in, i64 %i
  %x = load double, ptr %src, align 8
  %y = fmul double %x, %factor
  %dst = getelementptr inbounds double, ptr %out, i64 %i
  store double %y, ptr %dst, align 8
  %next = add nuw nsw i64 %i, 1
  %more = icmp slt i64 %next, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}