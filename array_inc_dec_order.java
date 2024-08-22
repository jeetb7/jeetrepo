package java_with_vscode.Practise_Basic_Code;
import java.util.*;
public class array_inc_dec_order {
    public static void main(String[] args) {
        int arr[]={7,8,9,1,6,5};
        int n=arr.length;
        Arrays.sort(arr);
        for(int i=0;i<n/2;i++){
            System.out.print(arr[i]+" ");
        }
        for(int i=n-1;i>=n/2;i--){
            System.out.print(arr[i]+" ");
        }
    }

}
